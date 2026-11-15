"""Praxis 公共 web 适配器:DNS 固定 HTTP 客户端 + URL 预检 + 逐请求限额(T10)。

只 import protocol + httpx/httpcore + stdlib(adapters 边界契约);core/permission
与 composition 的接线在 T11,工具值在 T12。
"""

from praxis.adapters.web.client import (
    CLIENT_ERROR_CODES,
    PublicHttpClient,
    PublicWebError,
    Resolver,
    WebFetchResponse,
    authorize_callback,
    validate_proxy_url,
    web_proxy_configuration,
)
from praxis.adapters.web.limits import WebLimits, load_web_limits
from praxis.adapters.web.urls import is_public_address, validate_public_url

__all__ = [
    "CLIENT_ERROR_CODES",
    "PublicHttpClient",
    "PublicWebError",
    "Resolver",
    "WebFetchResponse",
    "WebLimits",
    "authorize_callback",
    "is_public_address",
    "load_web_limits",
    "validate_proxy_url",
    "validate_public_url",
    "web_proxy_configuration",
]
