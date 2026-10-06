"""Bounded public-document extraction, preserving source text and navigable links."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urljoin, urlsplit
from uuid import uuid4

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

from agent_runtime.tools.web_http import PublicWebError, validate_public_url

MAX_SOURCE_CHARACTERS = 200_000
MAX_SOURCE_LINKS = 200
_PARSER_TIMEOUT_SECONDS = 10.0


async def extract_content_async(body: bytes, content_type: str, url: str) -> ExtractedContent:
    """Parse untrusted document bytes in a killable worker with no inherited secrets."""
    result = await _parse_in_worker(body, content_type, url)
    return ExtractedContent(**{**result, "links": tuple(result["links"])})


async def extract_search_results_async(body: bytes) -> list[dict[str, str]]:
    """Parse the fixed public Bing result page with the same worker limits."""
    result = await _parse_in_worker(body, "text/html", "https://www.bing.com/search", search=True)
    return cast(list[dict[str, str]], result["results"])


async def _parse_in_worker(
    body: bytes, content_type: str, url: str, *, search: bool = False,
) -> dict[str, Any]:
    if len(body) > 2_000_000:
        raise PublicWebError("web_response_too_large", "Document exceeds the parsing byte limit.")
    if content_type.split(";", 1)[0].strip().lower() == "application/pdf" and sys.platform != "linux":
        raise PublicWebError("web_content_unsupported", "PDF parsing requires Linux worker memory limits.")
    root = str(Path(__file__).resolve().parents[2])
    code = f"import sys; sys.path.insert(0, {root!r}); from agent_runtime.tools.web_content import _worker; _worker()"
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-B", "-c", code, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, cwd="/",
    )
    payload = json.dumps({"body": base64.b64encode(body).decode(), "content_type": content_type,
                          "url": url, "search": search}).encode()
    try:
        async with asyncio.timeout(_PARSER_TIMEOUT_SECONDS):
            output, _ = await process.communicate(payload)
        if process.returncode or len(output) > 2_000_000:
            raise PublicWebError("web_content_invalid", "Document parser failed or exceeded its resource limit.")
        result: dict[str, Any] = json.loads(output)
        if "error_code" in result:
            raise PublicWebError(result["error_code"], result["error_message"])
        return result
    except TimeoutError:
        raise PublicWebError("web_content_timeout", "Document parsing exceeded its time limit.") from None
    except (ValueError, KeyError):
        raise PublicWebError("web_content_invalid", "Document parser returned an invalid result.") from None
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.shield(process.wait())


def _worker() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    output: dict[str, Any]
    try:
        value = json.loads(sys.stdin.buffer.read(3_000_000))
        body = base64.b64decode(value["body"], validate=True)
        if value.get("search"):
            output = {"results": extract_search_results(body)}
        else:
            output = asdict(extract_content(body, value["content_type"], value["url"]))
    except PublicWebError as error:
        output = {"error_code": error.code, "error_message": str(error)}
    except Exception:
        output = {"error_code": "web_content_invalid", "error_message": "Document parsing failed."}
    sys.stdout.write(json.dumps(output, ensure_ascii=False))


def extract_search_results(body: bytes) -> list[dict[str, str]]:
    """Return organic results only; challenge/unknown pages are failures, never empty success."""
    soup = BeautifulSoup(body, "html.parser")
    if soup.select_one('#b_captcha, #captcha, iframe[src*="captcha"], form[action*="challenge"]'):
        raise PublicWebError("web_search_blocked", "Search service requires verification; do not retry this Turn.")
    entries = soup.select("#b_results li.b_algo")
    if not entries:
        if soup.select_one("#b_results .b_no"):
            return []
        raise PublicWebError("web_search_invalid_response", "Search service returned an unrecognized page.")
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries[:20]:
        anchor = entry.select_one("h2 a[href]")
        if anchor is None:
            continue
        href = str(anchor.get("href", ""))
        if len(href) > 8192:
            continue
        try:
            parsed = urlsplit(urljoin("https://www.bing.com", href))
            if parsed.hostname in {"bing.com", "www.bing.com", "cn.bing.com"} and parsed.path == "/ck/a":
                encoded = parse_qs(parsed.query).get("u", [""])[0]
                if not encoded.startswith("a1"):
                    continue
                encoded = encoded[2:]
                href = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True).decode()
            url = str(validate_public_url(href))
            if len(url) > 4096 or urlsplit(url).hostname in {"bing.com", "www.bing.com", "cn.bing.com"}:
                continue
        except (PublicWebError, ValueError, UnicodeError):
            continue
        if url in seen:
            continue
        seen.add(url)
        caption = entry.select_one(".b_caption p")
        results.append({"title": anchor.get_text(" ", strip=True)[:500], "url": url,
                        "snippet": caption.get_text(" ", strip=True)[:2000] if caption else ""})
    if not results:
        raise PublicWebError("web_search_invalid_response", "Search page contained no valid public result links.")
    return results


@dataclass(frozen=True, slots=True)
class ExtractedContent:
    title: str
    text: str
    links: tuple[dict[str, str], ...] = ()
    truncated: bool = False
    published_at: str | None = None
    published_at_source: str | None = None
    extraction_method: str = "text"
    link_occurrences: tuple[dict[str, int], ...] = ()
    links_truncated: bool = False


def extract_content(
    body: bytes, content_type: str, url: str, *, max_characters: int = MAX_SOURCE_CHARACTERS,
) -> ExtractedContent:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not 1 <= max_characters <= MAX_SOURCE_CHARACTERS:
        raise ValueError("invalid content character limit")
    title = url.rsplit("/", 1)[-1] or "Public source"
    links: tuple[dict[str, str], ...] = ()
    truncated = False
    published_at = published_at_source = None
    extraction_method = "text"
    occurrences: tuple[dict[str, int], ...] = ()
    links_truncated = False
    if media_type == "application/pdf":
        import pymupdf

        try:
            with pymupdf.open(stream=body, filetype="pdf") as document:  # type: ignore[no-untyped-call]
                if document.needs_pass:
                    raise PublicWebError("web_content_unsupported", "Password-protected PDFs are unsupported.")
                chunks: list[str] = []
                length = 0
                for index in range(min(len(document), 20)):
                    page_text = document[index].get_text()
                    if not page_text.strip():
                        continue
                    text = f"Page {index + 1}\n" + page_text
                    chunks.append(text[:max_characters - length])
                    length += len(text)
                    if length >= max_characters:
                        truncated = True
                        break
                truncated = truncated or len(document) > 20
                text = "\n\n".join(chunks)
        except PublicWebError:
            raise
        except Exception:
            raise PublicWebError("web_content_invalid", "The PDF could not be parsed.") from None
    elif media_type in {"text/html", "application/xhtml+xml"}:
        soup = BeautifulSoup(body, "html.parser")
        nonce = uuid4().hex
        for node in soup.find_all(True):
            node.attrs.pop("data-praxis-link", None)
            node.attrs.pop("data-praxis-selected", None)
        published_at, published_at_source = _published_time(soup)
        if soup.title:
            title = soup.title.get_text(" ", strip=True)[:500]
        for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
            comment.extract()
        for element in soup.find_all([
            "head", "script", "style", "noscript", "template", "nav", "footer", "aside",
            "form", "dialog", "svg", "button", "input",
        ]):
            element.decompose()
        for element in list(soup.find_all(["div", "section", "ul", "header"])):
            if element.parent is None:
                continue
            classes = element.get("class")
            class_text = " ".join(classes) if isinstance(classes, list) else str(classes or "")
            identity = " ".join([str(element.get("id", "")), class_text,
                                 str(element.get("role", ""))])
            if _NOISE.search(re.sub(r"([a-z])([A-Z])", r"\1 \2", identity)):
                element.decompose()
        root, extraction_method = _content_root(soup)
        if extraction_method in {"paragraph_density", "trafilatura"}:
            for paragraph in list(root.find_all("p")):
                text_length = len(paragraph.get_text(strip=True))
                linked = sum(len(a.get_text(strip=True)) for a in paragraph.find_all("a"))
                if text_length > 20 and linked / text_length > 0.9:
                    paragraph.decompose()
        if not root.get_text(strip=True):
            raise PublicWebError("web_content_empty", "No readable text was found in the HTML body.")
        found: list[dict[str, str]] = []
        terms = [term.casefold() for term in re.findall(r"[^\W_]+", title) if len(term) >= 4]
        related = [anchor for anchor in soup.find_all("a", href=True)
                   if not any(parent is root for parent in anchor.parents)
                   and not any(parent.get("data-praxis-selected") for parent in anchor.parents
                               if isinstance(parent, Tag))
                   and len(anchor.get_text(strip=True)) >= 8
                   and any(term in anchor.get_text(" ", strip=True).casefold() for term in terms)][:20]
        related_labels: list[str] = []
        for anchor in [*root.find_all("a", href=True), *related]:
            try:
                href = str(validate_public_url(urljoin(url, str(anchor.get("href")))))
            except (PublicWebError, ValueError):
                continue
            if len(href) > 4096:
                continue
            label = anchor.get_text(" ", strip=True)[:200]
            if not label:
                label = " ".join(str(image.get("alt", "")) for image in anchor.find_all("img"))[:200]
            if not label.strip():
                continue
            if len(found) < MAX_SOURCE_LINKS:
                found.append({"url": href, "text": label})
                anchor["data-praxis-link"] = f"{nonce}:{len(found)}"
                if any(anchor is related_anchor for related_anchor in related):
                    related_labels.append(label)
            else:
                links_truncated = True
        links = tuple(found)
        text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", _html_text(root, url)).strip()
        if related_labels:
            text += ("\n\n## Related link labels (article text not yet verified)\n"
                     + "\n".join("- " + _html_text(anchor, url).strip() for anchor in related
                                 if anchor.get("data-praxis-link")))
        # Only renderer-generated boundaries determine positions. Page text and
        # code containing literal [L1] cannot impersonate an actual hyperlink.
        marker_pattern = re.compile(r"\x00" + nonce + r":(\d+)\x00(.*?)\x00/" + nonce + r":\1\x00", re.S)
        pieces: list[str] = []
        located: list[dict[str, int]] = []
        cursor = size = 0
        for match in marker_pattern.finditer(text):
            prefix, rendered = text[cursor:match.start()], match[2]
            pieces.extend((prefix, rendered))
            size += len(prefix)
            located.append({"link_index": int(match[1]) - 1, "start": size, "end": size + len(rendered)})
            size += len(rendered)
            cursor = match.end()
        pieces.append(text[cursor:])
        text, occurrences = "".join(pieces), tuple(located)
    elif media_type.startswith("text/") or media_type in {
        "application/json", "application/ld+json", "application/xml", "application/javascript",
    }:
        if b"\x00" in body:
            raise PublicWebError("web_content_unsupported", "Binary content is unsupported.")
        try:
            text = body.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise PublicWebError("web_content_unsupported", "Text documents must use UTF-8 encoding.") from None
        if media_type in {"application/json", "application/ld+json"}:
            try:
                json.loads(text)
            except (ValueError, RecursionError):
                raise PublicWebError("web_content_invalid", "The JSON document is invalid.") from None
        if (media_type in {"text/markdown", "text/x-markdown"}
                or urlsplit(url).path.lower().endswith((".md", ".markdown"))):
            links, occurrences, links_truncated = _markdown_links(text, url)
    else:
        raise PublicWebError("web_content_unsupported", "This response content type is unsupported.")
    if not text.strip():
        raise PublicWebError(
            "web_content_empty", "No readable text was found; the page may require JavaScript or authentication.",
        )
    return ExtractedContent(title[:500], text[:max_characters], links, truncated or len(text) > max_characters,
                            published_at, published_at_source, extraction_method,
                            tuple(item for item in occurrences if item["start"] < max_characters), links_truncated)


def _html_text(node: Tag | NavigableString, url: str) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return " ".join(str(node).split()) + (" " if str(node).strip() else "")
    if node.name == "pre":
        code = node.find("code")
        language = ""
        if isinstance(code, Tag):
            classes = code.get("class")
            for value in classes if isinstance(classes, list) else []:
                if re.fullmatch(r"language-[\w+-]{1,30}", value):
                    language = value.removeprefix("language-")
                    break
        return "\n```" + language + "\n" + node.get_text().rstrip("\n") + "\n```\n"
    if node.name == "img":
        return str(node.get("alt", "")).strip() + " "
    if node.name == "br":
        return "\n"
    if node.name == "a":
        label = "".join(_html_text(child, url) for child in node.children
                        if isinstance(child, (Tag, NavigableString))).strip()
        marker = node.get("data-praxis-link")
        if marker:
            identifier = str(marker).rsplit(":", 1)[1]
            return f"\x00{marker}\x00[{label}][L{identifier}]\x00/{marker}\x00 "
        return label + " "
    content = "".join(_html_text(child, url) for child in node.children if isinstance(child, (Tag, NavigableString)))
    if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "\n" + "#" * int(node.name[1]) + " " + content.strip() + "\n"
    if node.name == "li":
        return "\n- " + content.strip() + "\n"
    if node.name in {
        "p", "div", "section", "article", "main", "tr", "ul", "ol", "table", "blockquote", "dl", "dt", "dd",
    }:
        return "\n" + content.strip() + "\n"
    if node.name in {"td", "th"}:
        return content.strip() + "\t"
    return content


# Generic DOM roles and whole class/id words, never site names or task keywords.
_NOISE = re.compile(
    r"(?:^|[-_\s])(?:nav|navigation|menu|sidebar|footer|advertisement|advert|ads|social|share|sharing|"
    r"recommend|recommended|recommendations|related|comments|comment|cookie|banner|breadcrumb|qr|qrcode)(?:$|[-_\s])",
    re.I,
)


def _content_root(soup: BeautifulSoup) -> tuple[Tag, str]:
    """Prefer semantic document regions; otherwise select the densest paragraph container."""
    semantic = soup.select("main, [role=main], article, [itemprop=articleBody]")
    if semantic:
        root = max(semantic, key=lambda node: len(node.get_text(strip=True)))
        articles = root.select("article, [itemprop=articleBody]")
        # A single self-contained article can have surrounding application chrome.
        # Multiple articles and directories remain one document, preserving their siblings.
        if len(articles) == 1:
            outside = [node for node in root.find_all(["a", "p", "pre", "table"])
                       if not any(p is articles[0] for p in node.parents)
                       and (node.name != "p" or bool(node.get_text(strip=True)))]
            if not outside:
                root = articles[0]
            else:
                # Read the primary document first, retaining meaningful sibling
                # data and directory links later in this same immutable view.
                root.insert(0, articles[0].extract())
        return root, "semantic"
    candidates: dict[int, tuple[Tag, float]] = {}
    for paragraph in soup.find_all(["p", "pre"]):
        length = len(paragraph.get_text(strip=True))
        if length < 40:
            continue
        parent = paragraph.parent
        for depth in range(3):
            if not isinstance(parent, Tag) or parent.name in {"body", "html", "[document]"}:
                break
            key = id(parent)
            old = candidates.get(key, (parent, 0.0))[1]
            candidates[key] = (parent, old + length / (depth + 1))
            parent = parent.parent
    if candidates:
        def score(candidate: tuple[Tag, float]) -> float:
            node, paragraph_score = candidate
            total = max(1, len(node.get_text(strip=True)))
            linked = sum(len(a.get_text(strip=True)) for a in node.find_all("a"))
            return paragraph_score * (1 - min(1, linked / total))
        root = max(candidates.values(), key=score)[0]
        root["data-praxis-selected"] = "1"
        # Restrict article extraction to the structural candidate; whole-page
        # extraction can reintroduce widgets and silently lose document sections.
        mature = _article_root(root)
        if mature is not None:
            evidence = [re.sub(r"\s+", "", p.get_text()) for p in root.find_all(["p", "pre"])
                        if len(p.get_text(strip=True)) >= 40]
            extracted = re.sub(r"\s+", "", mature.get_text())
            if all(block in extracted for block in evidence):
                return mature, "trafilatura"
        return root, "paragraph_density"
    return soup.body or soup, "body_fallback"


def _article_root(soup: Tag) -> Tag | None:
    """Use a mature article extractor on cleaned DOM; keep the original fallback available."""
    from trafilatura import extract

    if len(soup.get_text(strip=True)) < 400:
        return None
    xml = extract(str(soup), output_format="xml", include_comments=False, include_links=True,
                  include_tables=True, include_images=True, include_formatting=True, favor_recall=True)
    if not xml:
        return None
    document = BeautifulSoup(xml, "xml")
    root = document.find("main")
    if not isinstance(root, Tag):
        return None
    for node in root.find_all(True):
        if node.name == "ref":
            node.name = "a"
            node["href"] = str(node.get("target") or "")
        elif node.name == "head":
            level = str(node.get("rend", "h2"))
            node.name = level if level in {"h1", "h2", "h3", "h4", "h5", "h6"} else "h2"
        elif node.name == "graphic":
            node.name = "img"
        elif node.name == "code":
            node.name = "pre"
        elif node.name in {"row", "cell", "list", "item"}:
            node.name = {"row": "tr", "cell": "td", "list": "ul", "item": "li"}[node.name]
    return root


def _markdown_links(text: str, url: str) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, int], ...], bool]:
    from markdown_it import MarkdownIt
    from markdown_it.rules_inline import StateInline
    from markdown_it.rules_inline import link as parse_link

    def located_link(state: StateInline, silent: bool) -> bool:
        start, count = state.pos, len(state.tokens)
        matched = parse_link(state, silent)
        if matched and not silent:
            token = next((item for item in state.tokens[count:] if item.type == "link_open"), None)
            if token is not None:
                token.meta["source_span"] = (start, state.pos)
        return matched

    parser = MarkdownIt("commonmark")
    parser.inline.ruler.at("link", located_link)

    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    links: list[dict[str, str]] = []
    occurrences: list[dict[str, int]] = []
    truncated = False
    for block in parser.parse(text):
        if block.type != "inline" or block.map is None:
            continue
        start, end = offsets[block.map[0]], offsets[block.map[1]]
        # Inline content may omit list/blockquote prefixes. Map parser positions
        # back to the unchanged original rather than searching repeated labels.
        positions: list[int] = []
        cursor = start
        for fragment in block.content.splitlines(keepends=True):
            part = fragment.rstrip("\r\n")
            position = text.find(part, cursor, end)
            if position < 0:
                position = cursor
            positions.extend(range(position, position + len(part)))
            if fragment.endswith("\n"):
                positions.append(position + len(part))
            cursor = position + len(part)
        for index, token in enumerate(block.children or []):
            if token.type != "link_open":
                continue
            try:
                target = str(validate_public_url(urljoin(url, str(token.attrGet("href") or ""))))
            except (PublicWebError, ValueError):
                continue
            if len(links) >= MAX_SOURCE_LINKS:
                truncated = True
                continue
            label = ""
            for child in (block.children or [])[index + 1:]:
                if child.type == "link_close":
                    break
                label += child.content
            links.append({"url": target, "text": label[:200]})
            span_start, span_end = token.meta["source_span"]
            if span_end <= len(positions):
                occurrences.append({"link_index": len(links) - 1, "start": positions[span_start],
                                    "end": positions[span_end - 1] + 1})
    return tuple(links), tuple(occurrences), truncated


def _published_time(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """Return a declared publication date, never infer one from fetch time or arbitrary text."""
    for key in ("article:published_time", "datePublished", "pubdate", "publishdate"):
        meta = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if isinstance(meta, Tag) and (value := _date_value(meta.get("content"))):
            return value, "meta:" + key
    for script in soup.select('script[type="application/ld+json"]')[:20]:
        try:
            payload = json.loads(script.get_text())
        except (ValueError, RecursionError):
            continue
        pending = [payload]
        visited = 0
        while pending and visited < 200:
            value = pending.pop()
            visited += 1
            if isinstance(value, dict):
                if date := _date_value(value.get("datePublished")):
                    return date, "jsonld:datePublished"
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    time = soup.select_one('time[itemprop="datePublished"][datetime], article time[datetime]')
    if time and (date := _date_value(time.get("datetime"))):
        return date, "time:datetime"
    return None, None


def _date_value(value: object) -> str | None:
    from datetime import datetime

    if not isinstance(value, str) or len(value) > 80:
        return None
    value = value.strip()
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value
