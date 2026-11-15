"""有界 HTML→文本抽取(stdlib ``html.parser``;agent_runtime/tools/web_content.py 语义移植)。

只保留 T12 需要的部分:**无可导航链接清单**(有意差异:旧 ``links``/``view=links``
面不移植——超限回读由管道外部化 + ``read_artifact`` 覆盖,分析用 ``search_text``),
无 PDF、无 markdownify、无 worker 子进程。解析全程只用标准库,不新增 bs4/lxml 依赖。

阈值与语义(旧 ``web_content.py`` 对照):
- ``MAX_SOURCE_CHARACTERS = 200_000``:先对**解码后的源 HTML/文本**截断再解析,
  解析结果再按同一上限截断;``ExtractedText.truncated`` 如实标记任何一侧被截。
- 噪声标签整段丢弃:``script``/``style``/``template``/``noscript``/``svg``
  (旧实现丢弃 head/script/style/template)+ ``nav``/``aside``(控制器裁决的点名噪声)。
  ``head`` 单独处理:HTML 允许 ``<head>`` 无闭合地进入 ``<body>``,按深度计会吞掉
  整篇文档,因此遇到 ``<body>`` 即结束 head 跳过态。
- 块级标签(``p``/``div``/``h1..h6``/``li``/``tr``/``pre`` 等)与 ``br``/``hr`` 产生
  换行;空白折叠是**全文级**的(含 ``pre`` 内部的缩进):本模块只承诺"块边界可读",
  不承诺保留代码缩进——T12 的行窗口与旧工具同样按虚拟行编号。
- 非 HTML 文本(content-type ``text/*`` 或 ``application/{json,ld+json,xml,javascript}``)
  按声明的 charset(缺省 UTF-8)宽松解码(``errors="replace"``),含 NUL 视为二进制
  拒收;其余 content-type 一律 ``web_content_unsupported``。
- 错误码(PublicWebError,由 tools.py 统一映射为 ``web_fetch_failed:<code>``):
  ``web_content_unsupported``(二进制/未知类型)、``web_content_empty``(无可读文本)。

**有意差异(解析超时)**:旧实现把解析放进可 SIGKILL 的 worker 子进程(10s + RLIMIT);
本移植**在进程内线程**执行(``asyncio.to_thread``),只受管道工具级 75s 预算约束。
线程不可杀——超时/取消后解析必须自然结束(tools.py 因此声明
``InterruptBehavior.FINISH_CURRENT``,与 3A inspect_data_file 同一姿态)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from praxis.adapters.web.urls import PublicWebError

__all__ = [
    "MAX_SOURCE_CHARACTERS",
    "MAX_VIRTUAL_LINE_CHARACTERS",
    "ExtractedText",
    "extract_text",
    "is_html_content_type",
    "media_type",
    "source_lines",
]

# 旧 web_content.py:24 逐字。
MAX_SOURCE_CHARACTERS = 200_000
# 旧 builtins/web.py:_source_lines 的 512 字符虚拟行宽(单行巨型段落仍可翻页)。
MAX_VIRTUAL_LINE_CHARACTERS = 512

# 整段丢弃的噪声标签(见模块 docstring;nav/aside 是控制器点名的行为表差异)。
_NOISE_TAGS = frozenset(
    {"script", "style", "template", "noscript", "svg", "nav", "aside"}
)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_TEXT_MEDIA_PREFIXES = ("text/",)
_TEXT_MEDIA_TYPES = frozenset(
    {"application/json", "application/ld+json", "application/xml", "application/javascript"}
)
_HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_CHARSET_RE = re.compile(r'charset\s*=\s*"?([A-Za-z0-9._:-]+)"?', re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_LINE_SPACE_RE = re.compile(r" *\n *")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class ExtractedText:
    """抽取结果;``truncated`` = 源或抽取文本触及 200_000 字符上限。"""

    text: str
    truncated: bool = False


def media_type(content_type: str) -> str:
    """``Content-Type`` 头 → 小写媒体类型(去参数;缺省空串)。"""
    return content_type.split(";", 1)[0].strip().lower()


def is_html_content_type(content_type: str) -> bool:
    """与旧实现同一集合:text/html 与 application/xhtml+xml(含参数)。"""
    return media_type(content_type) in _HTML_MEDIA_TYPES


def _declared_charset(content_type: str) -> str | None:
    match = _CHARSET_RE.search(content_type)
    return match[1] if match is not None else None


def _decode_body(body: bytes, content_type: str) -> str:
    """宽松解码:声明 charset 优先,其次 UTF-8;非法字节替换而非抛错。"""
    charset = _declared_charset(content_type) or "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        # 未知/恶意 charset 名:回到 UTF-8,仍是宽松解码。
        return body.decode("utf-8", errors="replace")


def source_lines(text: str) -> list[str]:
    """稳定虚拟行:真实行按 512 字符切块;空行保留为空串(旧 ``_source_lines`` 同款)。

    行号用于行窗口游标;巨型单行段落因此也永远能翻页。
    """
    lines: list[str] = []
    for line in text.splitlines():
        if line:
            lines.extend(
                line[index : index + MAX_VIRTUAL_LINE_CHARACTERS]
                for index in range(0, len(line), MAX_VIRTUAL_LINE_CHARACTERS)
            )
        else:
            lines.append("")
    return lines


class _TextExtractor(HTMLParser):
    """把 HTML 折叠为"块边界有换行"的纯文本;噪声整段跳过。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._noise_depth = 0
        # head 可能无闭合地进入 body:单独用布尔态,遇到 <body> 立即解除。
        self._in_head = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _NOISE_TAGS:
            self._noise_depth += 1
            return
        if self._noise_depth:
            return
        if tag == "head":
            self._in_head = True
            return
        if self._in_head:
            if tag == "body":
                self._in_head = False
            else:
                return
        if tag in _BLOCK_TAGS or tag == "br":
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # 自闭合块(如 <hr/>):只产一个换行,不进入配对计数。
        if tag in _NOISE_TAGS or self._noise_depth or self._in_head:
            return
        if tag in _BLOCK_TAGS or tag == "br":
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _NOISE_TAGS:
            if self._noise_depth:
                self._noise_depth -= 1
            return
        if self._noise_depth:
            return
        if tag == "head":
            self._in_head = False
            return
        if self._in_head:
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._noise_depth or self._in_head:
            return
        self._parts.append(data.replace("\xa0", " "))

    def text(self) -> str:
        return _normalize("".join(self._parts))


def _normalize(raw: str) -> str:
    """空白折叠 + 块边界保留(旧 markdownify 输出的可读性下限)。"""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _LINE_SPACE_RE.sub("\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _html_to_text(source: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(source)
        parser.close()
    except Exception:
        # HTMLParser 对畸形输入极其宽容;若仍炸,按"已解析部分"交付而非炸穿 turn。
        pass
    return parser.text()


def extract_text(body: bytes, content_type: str) -> ExtractedText:
    """有界抽取:HTML → 文本;受支持的非 HTML → 宽松解码文本;其余拒收。

    源与产物各受 ``MAX_SOURCE_CHARACTERS`` 约束;``truncated`` 覆盖两侧。
    """
    kind = media_type(content_type)
    if kind in _HTML_MEDIA_TYPES:
        decoded = _decode_body(body, content_type)
        truncated = len(decoded) > MAX_SOURCE_CHARACTERS
        text = _html_to_text(decoded[:MAX_SOURCE_CHARACTERS])
        if len(text) > MAX_SOURCE_CHARACTERS:
            text = text[:MAX_SOURCE_CHARACTERS]
            truncated = True
    elif kind.startswith(_TEXT_MEDIA_PREFIXES) or kind in _TEXT_MEDIA_TYPES:
        if b"\x00" in body:
            raise PublicWebError("web_content_unsupported", "Binary content is unsupported.")
        decoded = _decode_body(body, content_type)
        truncated = len(decoded) > MAX_SOURCE_CHARACTERS
        text = decoded[:MAX_SOURCE_CHARACTERS]
    else:
        raise PublicWebError("web_content_unsupported", "This response content type is unsupported.")
    if not text.strip():
        raise PublicWebError("web_content_empty", "No extracted text was found.")
    return ExtractedText(text=text, truncated=truncated)