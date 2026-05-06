"""Custom exceptions for Kiji Inspector deployer"""


class KijiServiceError(Exception):
    """Base exception for Kiji service operations"""
    pass


class ClusterNotRunningError(Exception):
    """Raised when the Kubernetes cluster is not running or unreachable"""
    pass


class ActionNotDefinedError(Exception):
    """Raised when a macro action is not defined or not selected"""
    pass
