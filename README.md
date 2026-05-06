# Kiji Inspector Plugin

A Dataiku DSS plugin for deploying and managing Kiji Inspector models with activation layer inspection capabilities on Kubernetes clusters.

## Overview

The Kiji Inspector Plugin enables users to:
- Deploy VLLM models with activation inspection capabilities to Kubernetes clusters
- Manage deployed services (list, inspect, remove)
- Use a custom LLM agent that captures and persists activation explanations to a database
- Analyze model behavior through Sparse Autoencoders (SAE) feature explanations

## Features

### 1. Kiji Inspector Deployer (Macro)

Deploy and manage VLLM containers with activation layer inspection on Kubernetes clusters.

**Actions:**
- **Deploy**: Create a new Kiji service deployment
- **Inspect**: List and view details of deployed services
- **Remove**: Delete an existing Kiji service

**Configuration Options:**

#### Service Configuration
- **Service Name**: Unique identifier for the deployment (default: `kiji-vllm`)
- **Namespace**: Kubernetes namespace (default: `kiji-services`)

#### Container Configuration
- **Container Image**: Docker image for VLLM with activations support
- **Image Tag**: Version/tag of the container image

#### Model Configuration
- **Model Name**: HuggingFace model identifier (e.g., `google/gemma-4-E4B-it`)
- **Activation Layers**: Comma-separated layer indices to inspect (e.g., `8` or `8,16,24`)
- **Activation Explanation Top-K**: Number of top activations to explain (1-100)
- **Tensor Parallel Size**: Number of GPUs for tensor parallelism (1-8)
- **VLLM dtype**: Data type for model weights (auto, float16, bfloat16, float32)

#### Resource Configuration
- **Replicas**: Number of pod replicas (1-8)
- **GPU Devices**: Number of GPU devices per replica (1-8)
- **Use Fractional GPUs**: Enable Run:ai scheduler for fractional GPU allocation
  - GPU Fraction: Percentage of GPU per replica (e.g., 0.5 = 50%)
  - Dynamic GPU Fraction: Maximum resource consumption limit
- **CPU Request**: CPU allocation (e.g., `4` or `4000m`)
- **Memory Request**: Memory allocation (e.g., `16Gi`)
- **Node Selector**: Optional node label for pod placement (format: `key=value`)

#### Service Exposition
- **NodePort**: Expose service on a static port on each node (external access)
- **ClusterIP**: Internal-only access within the cluster
- **LoadBalancer**: Provision an external load balancer (if supported by cluster)
- **Service Port**: Port to expose the service (default: 8000)

### 2. Kiji Inspector Agent

A custom LLM agent that integrates with deployed Kiji Inspector services to capture activation explanations.

**Configuration:**
- **LLM Endpoint**: URL of the Kiji-compatible OpenAI API endpoint
- **Model Name**: Model identifier (must match the deployed service)
- **PostgreSQL Connection**: Database connection for persisting feature explanations
- **Features Table Name**: Table for storing activation explanations (default: `features_master`)

**Functionality:**
- Sends chat completion requests to the Kiji Inspector endpoint
- Extracts activation explanations from model responses
- Automatically creates and manages a PostgreSQL table with schema:
  - `timestamp_ms`: Request timestamp
  - `messages`: Input messages (JSON)
  - `settings`: LLM settings (JSON)
  - `response`: Complete response (JSON)
  - `layer_id`: Activation layer identifier
  - `feature_id`: Feature index
  - `description`: Human-readable feature explanation
  - `activation`: Activation strength value
- Provides tracing and logging for debugging

## Prerequisites

1. **Dataiku DSS** with Kubernetes cluster integration
2. **Kubernetes Cluster** with:
   - GPU support (NVIDIA GPUs)
   - Sufficient resources for VLLM workloads
   - Optional: Run:ai scheduler (for fractional GPU support)
3. **PostgreSQL Database** (for agent feature persistence)
4. **Container Registry** access to VLLM activation-enabled images

## Installation

1. Download or clone the plugin to your Dataiku instance
2. Install the plugin from the Dataiku Plugin Store or via the UI:
   - Administration → Plugins → Add Plugin
   - Select the plugin directory

## Usage

### Deploying a Kiji Service

1. Navigate to any Dataiku project
2. Go to **Macros** → **Kiji Inspector (VLLM Activations)**
3. Select your Kubernetes cluster
4. Choose **Action**: "Kiji Service: Deploy"
5. Configure your deployment parameters
6. Click **Run Macro**
7. Note the endpoint URL displayed in the results (you'll need this for the agent)

**Endpoint Format:**
- Pod Endpoint (recommended): `http://<pod-ip>:8000/v1/chat/completions`
- DNS Name: `http://<service-name>.<namespace>.svc.cluster.local:8000/v1/chat/completions`
- ClusterIP: `http://<cluster-ip>:8000/v1/chat/completions`
- NodePort: `http://<node-ip>:<nodeport>/v1/chat/completions`

### Using the Kiji Inspector Agent

1. Create or edit a Visual Agent
2. Select **Kiji Inspector Agent** as the LLM type
3. Configure the agent:
   - **LLM Endpoint**: Use the endpoint from the deployment step
   - **Model Name**: Must match the deployed model
   - **PostgreSQL Connection**: Select your database connection
   - **Features Table**: Specify table name for explanations
4. Use the agent in conversations or applications
5. Query the features table to analyze activation patterns

### Managing Services

**Inspect Services:**
1. Run the macro with Action: "Kiji Service: Inspect"
2. View all deployed services in the namespace with:
   - Service names and endpoints
   - Replica counts
   - Pod status
   - Available connection options

**Remove Services:**
1. Run the macro with Action: "Kiji Service: Remove"
2. Specify the service name to delete
3. Both deployment and service will be removed from the cluster

## Architecture

```
┌─────────────────────┐
│   Dataiku DSS       │
│                     │
│  ┌──────────────┐   │
│  │ Kiji Agent   │   │──────┐
│  └──────────────┘   │      │
└─────────────────────┘      │
                             │ HTTP Requests
                             │
┌────────────────────────────▼──────────────────────┐
│              Kubernetes Cluster                   │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  Kiji VLLM Service (Deployment)          │    │
│  │  ┌────────────────────────────────────┐  │    │
│  │  │  VLLM Container                    │  │    │
│  │  │  - Model Inference                 │  │    │
│  │  │  - Activation Layer Inspection     │  │    │
│  │  │  - SAE Feature Explanations        │  │    │
│  │  └────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  Service Endpoints:                               │
│  - NodePort, ClusterIP, or LoadBalancer           │
└───────────────────────────────────────────────────┘
                             │
                             │ Explanations
                             ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │  Features Table │
                    └─────────────────┘
```

## Troubleshooting

### Service Not Reachable

If the deployed service endpoint is not accessible:

1. **Wait for startup**: VLLM services can take several minutes to fully initialize
2. **Check pod status**: Run "Kiji Service: Inspect" to verify pods are running
3. **Try different endpoints**: The inspect results show multiple endpoint options
   - Start with the Pod Endpoint (most direct)
   - Try DNS name if inside the cluster
   - Use ClusterIP for internal access
4. **Verify network policies**: Ensure no NetworkPolicies are blocking traffic
5. **Check logs**: Use `kubectl logs -n <namespace> <pod-name>` to view container logs

### Agent Connection Issues

- Verify the endpoint URL is correct and includes `/v1/chat/completions`
- Ensure the model name in the agent matches the deployed model
- Check that the Dataiku instance can reach the Kubernetes network
- Test connectivity using `curl` from the Dataiku host

### Feature Table Issues

- Ensure the PostgreSQL connection is valid
- Verify the Dataiku service account has table creation permissions
- Check that the dataset is created in the correct project
- The table is auto-created on first agent use

## Examples

### Example 1: Deploy a Gemma Model with Layer 8 Inspection

```
Service Name: kiji-gemma
Model: google/gemma-4-E4B-it
Activation Layers: 8
Top-K: 5
GPUs: 1
Exposition: NodePort
```

### Example 2: Multi-layer Inspection with Run:ai

```
Service Name: kiji-multi-layer
Model: google/gemma-4-E4B-it
Activation Layers: 8,16,24
Top-K: 10
Use Fractional GPUs: Yes
GPU Fraction: 0.5
Dynamic GPU Fraction: 0.8
Exposition: ClusterIP
```

## License

Apache Software License

## Support

For issues, questions, or contributions, please contact the plugin author: shashank.gaur

## Version

Current version: 0.0.1
