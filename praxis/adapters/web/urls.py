"""公共 web URL 预检(移植 agent_runtime/tools/web_http.py:72-109;S6 主机名纪律)。

两段式的第一段:**URL 预检不做任何 DNS**(因此可在权限批准之前跑);主机名
到地址的裁决钉死在实际连接期(client.py 的 resolver 注入点)。主机名裁决一律
``urlsplit().hostname`` 精确/域名后缀匹配,**任何子串匹配禁止**——
``localhost.evil.com`` 不是 localhost,``github.com.evil.com`` 不匹配
``*.github.com``(后者由 protocol/policy._domain_matches 同一语义承担)。

``PublicWebError`` 定义在本模块(urls 与 client 共用、避免循环导入),client
再导出;fetch 路径自产码集见 client.py 模块 docstring,预检自产 ``invalid_url``
(legacy 逐字 parity,T12 映射为 invalid_input)。

IDNA 三层(R-T10-1):(a) stdlib ``host.encode("idna")`` 拒结构性非法(空标签/
超长标签;IDNA2003 scope,比 legacy 更严);(a') ``httpx.URL`` wire 语法门恢复
legacy IDNA2008 严格性(💩/ϐ 类在预检层拒);(b) client 另有 ``httpx.InvalidURL``
⇒ invalid_url 兜底。三层全零 DNS。
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import httpx

__all__ = ["PublicWebError", "is_public_address", "validate_public_url"]


class PublicWebError(RuntimeError):
    """稳定错误码 + 安全消息(不带远端数据、不带 URL 秘密)。

    ``code`` 是唯一机器可读分支面;消费者按 code 分支,绝不按消息文本。
    message 可省(T11 的 ``PublicWebError("denied")`` 形状)。
    """

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message)


def is_public_address(host: str) -> bool:
    """字面 IP 公网判定:IPv4-mapped IPv6 先还原,再 is_global 且非 multicast/reserved。

    非字面 IP(主机名)一律 False——主机名不在这里裁决,连接期逐答案校验。
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast and not address.is_reserved


# brief 点名的私有别名(legacy web_http.py:72 同名);单一实现仍是 is_public_address。
_is_public_address = is_public_address


def validate_public_url(url: str) -> tuple[str, int, str]:
    """只校验 URL 语法与字面地址,不做 DNS I/O;返回 (host, 有效端口, path+query 规范化串)。

    拒绝:scheme ∉ {http,https}、userinfo、端口 ∉ {None,80,443}、host 含 ``%``、
    ``localhost``/``.localhost`` 后缀、非公网字面 IP(IPv4-mapped 先还原)、
    空白/控制字符、IDNA 非法主机名(三层,见模块 docstring)。host 小写并去尾点;
    fragment 丢弃;空 path 规范化为 ``/``。任何失败抛 ``PublicWebError("invalid_url")``。
    """
    invalid = PublicWebError("invalid_url", "A public HTTP(S) URL on port 80 or 443 is required.")
    if not url or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url):
        raise invalid
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port  # 非法端口在这里抛 ValueError
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or port not in {None, 80, 443}
            or "%" in host
            or host == "localhost"
            or host.endswith(".localhost")
        ):
            raise invalid
        try:
            ipaddress.ip_address(host)
        except ValueError:
            # 主机名:预检不解析 DNS,连接期再逐答案裁决;但 IDNA 可编码性在这里判。
            # (a) stdlib idna 是 IDNA2003 scope:拒空标签/超长标签等结构性非法;
            # nameprep 会接受 💩/ϐ 之类 httpx(IDNA2008)拒绝的字符——那一类由
            # 下方 httpx.URL wire 门与 client 的 InvalidURL 兜底层负责。
            try:
                host.encode("idna")
            except UnicodeError:
                raise invalid from None
        else:
            if not is_public_address(host):
                raise invalid
        # (a') wire 语法门:恢复 legacy httpx.URL 预检的 IDNA2008 严格性(💩/ϐ 类
        # 在预检层拒)。只做语法裁决,不参与主机名判定(判定一律 urlsplit 上方已完成)。
        httpx.URL(url)
    except (ValueError, httpx.InvalidURL):
        raise invalid from None
    target = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
    effective_port = port if port is not None else (443 if parsed.scheme == "https" else 80)
    return host, effective_port, target
