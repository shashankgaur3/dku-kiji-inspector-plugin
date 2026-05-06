import dataiku
import logging
from dataiku.runnables import Runnable

import kiji_deployer.kiji_services as kiji_services

from kiji_deployer.utils import configure_kubeconfig
from kiji_deployer.exceptions import (
    ClusterNotRunningError,
    ActionNotDefinedError
)

import subprocess

LOGGER = logging.getLogger(__name__)


class MyRunnable(Runnable):
    """Kiji Inspector VLLM Deployer - Deploy VLLM containers with activation inspection"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config
        LOGGER.info("Kiji Inspector Runnable instantiated")
        LOGGER.debug(f"Configuration: {self.config}")

    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this function return a tuple of
        (target, unit) where unit is one of: SIZE, FILES, RECORDS, NONE
        """
        return None

    def run(self, progress_callback):
        """
        Performs the action selected in the 'macro_action' parameter.
        """
        LOGGER.info("Starting Kiji Inspector runnable execution.")

        # Unpack common parameters
        cluster_id = self.config.get("cluster_id", "")
        macro_action = self.config.get("macro_action")
        LOGGER.info(f"Executing macro action: '{macro_action}' on cluster '{cluster_id}'")

        # Unpack service configuration
        service_name = self.config.get("service_name", "kiji-vllm")
        namespace = self.config.get("namespace", "kiji-services")

        # Unpack container configuration
        container_image = self.config.get("container_image", "vllm-extras-activations")
        container_tag = self.config.get("container_tag", "latest")

        # Unpack model configuration
        model_name = self.config.get("model_name", "google/gemma-4-E4B-it")
        activation_layers = self.config.get("activation_layers", "8")
        activation_explanation_top_k = self.config.get("activation_explanation_top_k", 5)
        tensor_parallel_size = self.config.get("tensor_parallel_size", 1)
        vllm_dtype = self.config.get("vllm_dtype", "bfloat16")

        # Unpack resource configuration
        replicas = self.config.get("replicas", 1)
        num_gpu_devices = self.config.get("num_gpu_devices", 1)
        use_runai = self.config.get("use_runai", False)
        gpu_fraction = self.config.get("gpu_fraction")
        dynamic_gpu_fraction = self.config.get("dynamic_gpu_fraction")
        cpu_request = self.config.get("cpu_request", "4")
        memory_request = self.config.get("memory_request", "16Gi")
        node_selector = self.config.get("node_selector")

        # Unpack exposition configuration
        exposition_mode = self.config.get("exposition_mode", "nodeport")
        service_port = self.config.get("service_port", 8000)

        LOGGER.debug(f"Unpacked parameters: service_name='{service_name}', model_name='{model_name}', namespace='{namespace}'")

        # Set KUBECONFIG environment variable
        LOGGER.info(f"Configuring kubeconfig for cluster: {cluster_id}")
        configure_kubeconfig(cluster_id)

        # Check the cluster is running
        try:
            r = subprocess.run(["kubectl", "version"], capture_output=True)
            r.check_returncode()
            LOGGER.info("Kubernetes cluster is reachable.")
        except subprocess.CalledProcessError as err:
            raise ClusterNotRunningError(f"The Kubernetes cluster {cluster_id} is unreachable or not running.") from err

        # Perform macro action
        if macro_action == "kiji_service_add":
            r = kiji_services.add(
                namespace=namespace,
                service_name=service_name,
                container_image=container_image,
                container_tag=container_tag,
                model_name=model_name,
                activation_layers=activation_layers,
                activation_explanation_top_k=activation_explanation_top_k,
                tensor_parallel_size=tensor_parallel_size,
                vllm_dtype=vllm_dtype,
                replicas=replicas,
                num_gpu_devices=num_gpu_devices,
                use_runai=use_runai,
                gpu_fraction=gpu_fraction,
                dynamic_gpu_fraction=dynamic_gpu_fraction,
                cpu_request=cpu_request,
                memory_request=memory_request,
                node_selector=node_selector,
                exposition_mode=exposition_mode,
                service_port=service_port
            )
        elif macro_action == "kiji_service_list":
            r = kiji_services.list(
                namespace=namespace
            )
        elif macro_action == "kiji_service_rm":
            r = kiji_services.rm(
                namespace=namespace,
                service_name=service_name
            )
        else:
            raise ActionNotDefinedError("Macro action not selected; please select a macro action from the dropdown.")

        LOGGER.info(f"Action '{macro_action}' completed successfully.")
        LOGGER.debug(f"Result: {r}")
        return str(r)
