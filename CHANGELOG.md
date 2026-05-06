# Changelog

All notable changes to the Kiji Inspector Plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-05-05

### Added
- Kiji Inspector Deployer macro for managing VLLM deployments on Kubernetes
  - Deploy action with full model and resource configuration
  - Inspect action to view service details and endpoints
  - Remove action to delete services
- Kiji Inspector Agent for LLM interactions with activation explanation persistence
  - OpenAI-compatible API integration
  - Automatic PostgreSQL table creation and feature logging
- Support for standard GPU and Run:ai fractional GPU allocations
- Multiple service exposition modes (NodePort, ClusterIP, LoadBalancer)
- Enhanced endpoint discovery showing pod IPs, DNS names, and node addresses

### Fixed
- Module import errors (renamed directories from hyphens to underscores)
- Service port naming in Kubernetes manifests

### Known Limitations
- VLLM services may take several minutes to start up
- Requires GPU-enabled Kubernetes cluster
- Run:ai features require Run:ai scheduler

## [Unreleased]

### Planned
- Ingress support
- Health check configuration
- Multi-cluster management
- Service autoscaling
