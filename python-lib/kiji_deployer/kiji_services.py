import os
import time
import subprocess
import logging
import json

from .utils import run
from .exceptions import KijiServiceError

LOGGER = logging.getLogger(__name__)

KIJI_PORT = 8000

# Kubernetes Deployment YAML template
KIJI_DEPLOYMENT_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_name}
  namespace: {namespace}
  labels:
    app: {service_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {service_name}
  template:
    metadata:
      labels:
        app: {service_name}
      {annotations_yaml}
    spec:
      {node_selector_yaml}
      containers:
      - name: vllm
        image: {container_image}:{container_tag}
        ports:
        - containerPort: {service_port}
          protocol: TCP
        env:
        - name: MODEL_NAME
          value: "{model_name}"
        - name: ACTIVATION_LAYERS
          value: "{activation_layers}"
        - name: ACTIVATION_EXPLANATION_TOP_K
          value: "{activation_explanation_top_k}"
        - name: TENSOR_PARALLEL_SIZE
          value: "{tensor_parallel_size}"
        - name: VLLM_DTYPE
          value: "{vllm_dtype}"
        {env_vars_yaml}
        resources:
          requests:
            cpu: "{cpu_request}"
            memory: "{memory_request}"
          limits:
            {gpu_limits_yaml}
"""

# Run:ai deployment with fractional GPU
KIJI_DEPLOYMENT_RUNAI_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_name}
  namespace: {namespace}
  labels:
    app: {service_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {service_name}
  template:
    metadata:
      labels:
        app: {service_name}
      annotations:
        gpu-fraction: "{gpu_fraction}"
        {dynamic_gpu_yaml}
    spec:
      {node_selector_yaml}
      schedulerName: runai-scheduler
      containers:
      - name: vllm
        image: {container_image}:{container_tag}
        ports:
        - containerPort: {service_port}
          protocol: TCP
        env:
        - name: MODEL_NAME
          value: "{model_name}"
        - name: ACTIVATION_LAYERS
          value: "{activation_layers}"
        - name: ACTIVATION_EXPLANATION_TOP_K
          value: "{activation_explanation_top_k}"
        - name: TENSOR_PARALLEL_SIZE
          value: "{tensor_parallel_size}"
        - name: VLLM_DTYPE
          value: "{vllm_dtype}"
        {env_vars_yaml}
        resources:
          requests:
            cpu: "{cpu_request}"
            memory: "{memory_request}"
          limits:
            runai.io/gpu: {num_gpu_devices}
"""

# Kubernetes Service YAML templates
KIJI_SERVICE_NODEPORT_YAML = """
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {namespace}
spec:
  type: NodePort
  selector:
    app: {service_name}
  ports:
  - name: http
    port: {service_port}
    targetPort: {service_port}
    protocol: TCP
"""

KIJI_SERVICE_CLUSTERIP_YAML = """
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {namespace}
spec:
  type: ClusterIP
  selector:
    app: {service_name}
  ports:
  - name: http
    port: {service_port}
    targetPort: {service_port}
    protocol: TCP
"""

KIJI_SERVICE_LOADBALANCER_YAML = """
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {namespace}
spec:
  type: LoadBalancer
  selector:
    app: {service_name}
  ports:
  - name: http
    port: {service_port}
    targetPort: {service_port}
    protocol: TCP
"""

NODE_SELECTOR_YAML = """
      nodeSelector:
        {node_selector}"""

DYNAMIC_GPU_ANNOTATION = 'gpu-memory-limit: "{dynamic_gpu_fraction}"'

ENV_VARS_YAML = """
        - name: RUNAI_GPU_MEMORY_LIMIT
          value: "{dynamic_gpu_fraction}"
"""


def create_namespace_if_not_exists(namespace):
    """Create Kubernetes namespace if it doesn't exist"""
    LOGGER.info(f"Ensuring namespace '{namespace}' exists.")
    try:
        subprocess.run(
            ["kubectl", "get", "namespace", namespace],
            capture_output=True,
            check=True
        )
        LOGGER.info(f"Namespace '{namespace}' already exists.")
    except subprocess.CalledProcessError:
        LOGGER.info(f"Creating namespace '{namespace}'.")
        subprocess.run(
            ["kubectl", "create", "namespace", namespace],
            capture_output=True,
            check=True
        )


def add(
    namespace,
    service_name,
    container_image,
    container_tag,
    model_name,
    activation_layers,
    activation_explanation_top_k,
    tensor_parallel_size,
    vllm_dtype,
    replicas,
    num_gpu_devices,
    use_runai,
    gpu_fraction,
    dynamic_gpu_fraction,
    cpu_request,
    memory_request,
    node_selector,
    exposition_mode,
    service_port
):
    """
    Deploy a Kiji Inspector VLLM service to the Kubernetes cluster.
    """
    LOGGER.info(f"Deploying Kiji service '{service_name}' to namespace '{namespace}'.")

    # Ensure namespace exists
    create_namespace_if_not_exists(namespace)

    # Generate deployment YAML
    node_selector_yaml = ""
    if node_selector:
        node_selector_yaml = NODE_SELECTOR_YAML.format(node_selector=node_selector)

    env_vars_yaml = ""
    dynamic_gpu_yaml = ""
    annotations_yaml = ""

    if use_runai:
        # Use Run:ai deployment template
        if dynamic_gpu_fraction:
            dynamic_gpu_yaml = f'gpu-memory-limit: "{dynamic_gpu_fraction}"'
            env_vars_yaml = ENV_VARS_YAML.format(dynamic_gpu_fraction=dynamic_gpu_fraction)

        deployment_yaml = KIJI_DEPLOYMENT_RUNAI_YAML.format(
            service_name=service_name,
            namespace=namespace,
            replicas=replicas,
            container_image=container_image,
            container_tag=container_tag,
            service_port=service_port,
            model_name=model_name,
            activation_layers=activation_layers,
            activation_explanation_top_k=activation_explanation_top_k,
            tensor_parallel_size=tensor_parallel_size,
            vllm_dtype=vllm_dtype,
            gpu_fraction=gpu_fraction,
            num_gpu_devices=num_gpu_devices,
            dynamic_gpu_yaml=dynamic_gpu_yaml,
            env_vars_yaml=env_vars_yaml,
            cpu_request=cpu_request,
            memory_request=memory_request,
            node_selector_yaml=node_selector_yaml
        )
    else:
        # Use standard deployment template
        gpu_limits_yaml = f"nvidia.com/gpu: {num_gpu_devices}"

        deployment_yaml = KIJI_DEPLOYMENT_YAML.format(
            service_name=service_name,
            namespace=namespace,
            replicas=replicas,
            container_image=container_image,
            container_tag=container_tag,
            service_port=service_port,
            model_name=model_name,
            activation_layers=activation_layers,
            activation_explanation_top_k=activation_explanation_top_k,
            tensor_parallel_size=tensor_parallel_size,
            vllm_dtype=vllm_dtype,
            annotations_yaml=annotations_yaml,
            env_vars_yaml=env_vars_yaml,
            gpu_limits_yaml=gpu_limits_yaml,
            cpu_request=cpu_request,
            memory_request=memory_request,
            node_selector_yaml=node_selector_yaml
        )

    # Generate service YAML based on exposition mode
    if exposition_mode == "nodeport":
        service_yaml = KIJI_SERVICE_NODEPORT_YAML.format(
            service_name=service_name,
            namespace=namespace,
            service_port=service_port
        )
    elif exposition_mode == "clusterip":
        service_yaml = KIJI_SERVICE_CLUSTERIP_YAML.format(
            service_name=service_name,
            namespace=namespace,
            service_port=service_port
        )
    elif exposition_mode == "loadbalancer":
        service_yaml = KIJI_SERVICE_LOADBALANCER_YAML.format(
            service_name=service_name,
            namespace=namespace,
            service_port=service_port
        )
    else:
        raise KijiServiceError(f"Unknown exposition mode: {exposition_mode}")

    # Write YAML files to temporary location
    unix_timestamp = int(time.time())
    yaml_dir = f"/tmp/kiji-{unix_timestamp}"
    os.makedirs(yaml_dir, exist_ok=True)

    deployment_yaml_path = os.path.join(yaml_dir, "deployment.yaml")
    service_yaml_path = os.path.join(yaml_dir, "service.yaml")

    with open(deployment_yaml_path, "w") as f:
        f.write(deployment_yaml)
    with open(service_yaml_path, "w") as f:
        f.write(service_yaml)

    LOGGER.info(f"Generated YAML files in {yaml_dir}")
    LOGGER.debug(f"Deployment YAML:\n{deployment_yaml}")
    LOGGER.debug(f"Service YAML:\n{service_yaml}")

    # Apply deployment
    LOGGER.info("Applying Kubernetes deployment...")
    cmd = ["kubectl", "apply", "-f", deployment_yaml_path]
    err_msg = "Failed to apply Kiji deployment: {stderr}"
    run(cmd, err_msg, KijiServiceError)

    # Apply service
    LOGGER.info("Applying Kubernetes service...")
    cmd = ["kubectl", "apply", "-f", service_yaml_path]
    err_msg = "Failed to apply Kiji service: {stderr}"
    run(cmd, err_msg, KijiServiceError)

    # Get service details
    time.sleep(2)  # Wait for service to be created
    service_info = get_service_info(namespace, service_name)

    # Clean up temporary files
    try:
        os.remove(deployment_yaml_path)
        os.remove(service_yaml_path)
        os.rmdir(yaml_dir)
    except Exception as e:
        LOGGER.warning(f"Failed to clean up temporary files: {e}")

    result = f"""
    <h3>Kiji Inspector Service Deployed Successfully</h3>
    <p><strong>Service Name:</strong> {service_name}</p>
    <p><strong>Namespace:</strong> {namespace}</p>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Activation Layers:</strong> {activation_layers}</p>
    <p><strong>Replicas:</strong> {replicas}</p>
    <p><strong>GPU Devices:</strong> {num_gpu_devices}</p>
    {service_info}
    <hr>
    <p><em>Use 'Kiji Service: Inspect' to check deployment status.</em></p>
    """

    return result


def list(namespace):
    """
    List all Kiji services in the namespace.
    """
    LOGGER.info(f"Listing Kiji services in namespace '{namespace}'.")

    try:
        # Get deployments
        cmd = ["kubectl", "get", "deployments", "-n", namespace, "-o", "json"]
        result = subprocess.run(cmd, capture_output=True, check=True)
        deployments = json.loads(result.stdout.decode('utf-8'))

        # Get services
        cmd = ["kubectl", "get", "services", "-n", namespace, "-o", "json"]
        result = subprocess.run(cmd, capture_output=True, check=True)
        services = json.loads(result.stdout.decode('utf-8'))

        # Build HTML output
        html = f"<h3>Kiji Services in namespace '{namespace}'</h3>"

        if not deployments.get('items'):
            html += "<p>No deployments found in this namespace.</p>"
            return html

        html += "<table border='1' cellpadding='5' cellspacing='0'>"
        html += "<tr><th>Name</th><th>Replicas</th><th>Available</th><th>Image</th><th>Service Type</th><th>Endpoint</th></tr>"

        for deployment in deployments.get('items', []):
            name = deployment['metadata']['name']
            replicas = deployment['spec'].get('replicas', 0)
            available = deployment['status'].get('availableReplicas', 0)

            containers = deployment['spec']['template']['spec'].get('containers', [])
            image = containers[0]['image'] if containers else 'N/A'

            # Find matching service
            service_type = 'N/A'
            endpoint = 'N/A'
            for service in services.get('items', []):
                if service['metadata']['name'] == name:
                    service_type = service['spec'].get('type', 'N/A')

                    if service_type == 'NodePort':
                        ports = service['spec'].get('ports', [])
                        if ports:
                            node_port = ports[0].get('nodePort', 'N/A')
                            endpoint = f"NodePort: {node_port}"
                    elif service_type == 'LoadBalancer':
                        lb_ingress = service['status'].get('loadBalancer', {}).get('ingress', [])
                        if lb_ingress:
                            endpoint = lb_ingress[0].get('ip', lb_ingress[0].get('hostname', 'Pending'))
                    elif service_type == 'ClusterIP':
                        cluster_ip = service['spec'].get('clusterIP', 'N/A')
                        endpoint = f"ClusterIP: {cluster_ip}"
                    break

            html += f"<tr><td>{name}</td><td>{replicas}</td><td>{available}</td><td>{image}</td><td>{service_type}</td><td>{endpoint}</td></tr>"

        html += "</table>"
        return html

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        return f"<p>Error listing services: {error_msg}</p>"


def rm(namespace, service_name):
    """
    Remove a Kiji service from the Kubernetes cluster.
    """
    LOGGER.info(f"Removing Kiji service '{service_name}' from namespace '{namespace}'.")

    # Delete deployment
    LOGGER.info("Deleting deployment...")
    cmd = ["kubectl", "delete", "deployment", service_name, "-n", namespace]
    err_msg = "Failed to delete Kiji deployment: {stderr}"
    run(cmd, err_msg, KijiServiceError)

    # Delete service
    LOGGER.info("Deleting service...")
    cmd = ["kubectl", "delete", "service", service_name, "-n", namespace]
    err_msg = "Failed to delete Kiji service: {stderr}"
    run(cmd, err_msg, KijiServiceError)

    result = f"""
    <h3>Kiji Service Removed</h3>
    <p><strong>Service Name:</strong> {service_name}</p>
    <p><strong>Namespace:</strong> {namespace}</p>
    <p>The service and deployment have been successfully deleted.</p>
    """

    return result


def get_service_info(namespace, service_name):
    """
    Get information about a service endpoint.
    """
    try:
        cmd = ["kubectl", "get", "service", service_name, "-n", namespace, "-o", "json"]
        result = subprocess.run(cmd, capture_output=True, check=True)
        service = json.loads(result.stdout.decode('utf-8'))

        service_type = service['spec'].get('type', 'Unknown')
        info = f"<p><strong>Service Type:</strong> {service_type}</p>"

        if service_type == 'NodePort':
            ports = service['spec'].get('ports', [])
            if ports:
                node_port = ports[0].get('nodePort')
                port = ports[0].get('port')
                cluster_ip = service['spec'].get('clusterIP')
                info += f"<p><strong>NodePort:</strong> {node_port}</p>"
                info += f"<p><strong>Service Port:</strong> {port}</p>"
                info += f"<p><strong>ClusterIP:</strong> {cluster_ip}</p>"

                # Try to get node IPs
                try:
                    nodes_cmd = ["kubectl", "get", "nodes", "-o", "json"]
                    nodes_result = subprocess.run(nodes_cmd, capture_output=True, check=True)
                    nodes = json.loads(nodes_result.stdout.decode('utf-8'))

                    node_ips = []
                    for node in nodes.get('items', []):
                        for address in node['status'].get('addresses', []):
                            if address['type'] in ['InternalIP', 'ExternalIP']:
                                node_ips.append(f"{address['type']}: {address['address']}")

                    if node_ips:
                        info += f"<p><strong>Available Nodes:</strong></p><ul>"
                        for node_ip in node_ips:
                            info += f"<li>{node_ip}</li>"
                        info += "</ul>"
                except Exception as node_err:
                    LOGGER.warning(f"Could not fetch node IPs: {node_err}")

                # Get pod endpoints (same approach as NIM plugin)
                pod_endpoint = None
                try:
                    endpoints_cmd = ["kubectl", "get", "endpoints", service_name, "-n", namespace, "-o", "json"]
                    endpoints_result = subprocess.run(endpoints_cmd, capture_output=True, check=True)
                    endpoints = json.loads(endpoints_result.stdout.decode('utf-8'))

                    for subset in endpoints.get('subsets', []):
                        for address in subset.get('addresses', []):
                            pod_ip = address.get('ip')
                            for ep_port in subset.get('ports', []):
                                pod_port = ep_port.get('port')
                                pod_endpoint = f"{pod_ip}:{pod_port}"
                                break
                            if pod_endpoint:
                                break
                        if pod_endpoint:
                            break
                except Exception as ep_err:
                    LOGGER.warning(f"Could not fetch pod endpoint: {ep_err}")

                info += f"<p><strong>Agent Endpoint Options:</strong></p>"
                if pod_endpoint:
                    info += f"<p>1. <strong>Pod Endpoint (recommended):</strong> <code>http://{pod_endpoint}/v1/chat/completions</code></p>"
                info += f"<p>2. <strong>DNS Name:</strong> <code>http://{service_name}.{namespace}.svc.cluster.local:{port}/v1/chat/completions</code></p>"
                info += f"<p>3. <strong>ClusterIP:</strong> <code>http://{cluster_ip}:{port}/v1/chat/completions</code></p>"
                info += f"<p>4. <strong>NodePort:</strong> <code>http://&lt;node-ip&gt;:{node_port}/v1/chat/completions</code></p>"

        elif service_type == 'LoadBalancer':
            lb_ingress = service['status'].get('loadBalancer', {}).get('ingress', [])
            ports = service['spec'].get('ports', [])
            port = ports[0].get('port') if ports else 8000
            if lb_ingress:
                lb_ip = lb_ingress[0].get('ip', lb_ingress[0].get('hostname'))
                info += f"<p><strong>LoadBalancer IP:</strong> {lb_ip}</p>"
                info += f"<p><strong>Agent Endpoint:</strong> <code>http://{lb_ip}:{port}/v1/chat/completions</code></p>"
            else:
                info += "<p><em>LoadBalancer IP is pending...</em></p>"

        elif service_type == 'ClusterIP':
            cluster_ip = service['spec'].get('clusterIP')
            port = service['spec'].get('ports', [{}])[0].get('port')
            info += f"<p><strong>ClusterIP:</strong> {cluster_ip}:{port}</p>"
            info += f"<p><strong>DNS Name:</strong> {service_name}.{namespace}.svc.cluster.local</p>"
            info += f"<p><strong>Agent Endpoint:</strong> <code>http://{service_name}.{namespace}.svc.cluster.local:{port}/v1/chat/completions</code></p>"

        return info
    except Exception as e:
        LOGGER.warning(f"Failed to get service info: {e}")
        return "<p><em>Service information not available yet.</em></p>"
