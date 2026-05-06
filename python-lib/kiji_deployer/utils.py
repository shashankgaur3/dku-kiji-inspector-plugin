import os
import subprocess
import logging
import dataiku

logger = logging.getLogger(__name__)


def configure_kubeconfig(cluster_name):
    """
    Sets environment variable for kubeconfig of a cluster.

    Args:
        cluster_name: The name of the Dataiku cluster
    """
    client = dataiku.api_client()
    cluster = client.get_cluster(cluster_name)
    cluster_settings = cluster.get_settings().settings

    # Don't set if cluster is unmanaged (uses default kubeconfig location)
    if not cluster_settings["type"] == "manual":
        logger.info(f"Configuring KUBECONFIG for managed cluster '{cluster_name}'.")
        kubeconfig_path = cluster_settings["data"]["kube_config_path"]
        os.environ["KUBECONFIG"] = kubeconfig_path
    else:
        logger.info(f"Cluster '{cluster_name}' is unmanaged, using default kubeconfig.")


def run(cmd, err_msg="", exception=Exception):
    logger.debug(f"Running command: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True) 
        r.check_returncode()
    except subprocess.CalledProcessError as err:
        logger.error(f"Command failed. Stderr: {r.stderr.decode('utf-8')}")
        raise exception(err_msg.format(stderr=r.stderr.decode('utf-8'))) from err
        
    return r
