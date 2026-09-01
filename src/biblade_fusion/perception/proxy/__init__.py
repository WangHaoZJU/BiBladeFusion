"""Conservative geometry proxies derived from an initial blade observation."""

from biblade_fusion.perception.proxy.builder import (
    ProxyBuildError,
    build_bilateral_proxy,
)
from biblade_fusion.perception.proxy.model import BilateralBladeProxy
from biblade_fusion.perception.proxy.support import (
    PROXY_SUPPORT_ALGORITHM,
    ProxySupportError,
    ProxySupportSelection,
    select_proxy_support,
)

__all__ = [
    "PROXY_SUPPORT_ALGORITHM",
    "BilateralBladeProxy",
    "ProxyBuildError",
    "ProxySupportError",
    "ProxySupportSelection",
    "build_bilateral_proxy",
    "select_proxy_support",
]
