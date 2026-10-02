from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from agent_loco.sandbox import Workspace
from agent_loco.tools.base import ToolResult, ToolSpec, object_schema

SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
INSTANT_ENDPOINT = "https://api.duckduckgo.com/"
USER_AGENT = (
    "Mozilla/5.0 (compatible; agent-loco/0.1.0; +https://github.com/hhollenstain/agent-loco)"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en",
}
MAX_SEARCH_RESULTS = 8
DEFAULT_SEARCH_RESULTS = 5
MAX_FETCH_BYTES = 1_000_000
DEFAULT_FETCH_CHARS = 20_000
MAX_FETCH_CHARS = 40_000
SEARCH_TIMEOUT = 15.0
FETCH_TIMEOUT = 20.0
MAX_REDIRECTS = 4
_BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "metadata.goog",
}
_GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/"
    r"(?P<ref>[^/]+)/(?P<path>.+)$",
    re.IGNORECASE,
)
_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "nav",
    "footer",
    "form",
}
_HEADING_TAGS = {f"h{level}": "#" * level for level in range(1, 7)}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "ul",
    "ol",
    "tr",
    "br",
    "pre",
    "blockquote",
    "hr",
    *_HEADING_TAGS,
}


def web_tools(_workspace: Workspace) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="web_search",
            description=(
                "Search the public web for current documentation, APIs, architecture "
                "guidance, and best practices. Use this before inventing a library "
                "API, framework pattern, or recommendation that is not already in "
                "this repo. Returns titles, URLs, and snippets. Then call fetch_url "
                "on an official docs page."
            ),
            parameters=object_schema(
                {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Include the language, library, and version "
                            "when you know them."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return. Default 5, max 8.",
                    },
                },
                ["query"],
            ),
            handler=lambda query, max_results=DEFAULT_SEARCH_RESULTS: web_search(
                query, max_results
            ),
        ),
        ToolSpec(
            name="fetch_url",
            description=(
                "Fetch a public http(s) URL and return readable text. Use this to "
                "read official documentation after web_search. HTML becomes text; "
                "JSON is pretty-printed. Localhost and private-network URLs are "
                "rejected."
            ),
            parameters=object_schema(
                {
                    "url": {
                        "type": "string",
                        "description": "Public http or https URL to read.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": ("Maximum characters of text to return. Default 20000."),
                    },
                },
                ["url"],
            ),
            handler=lambda url, max_chars=DEFAULT_FETCH_CHARS: fetch_url(url, max_chars),
        ),
    ]


def web_search(query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> ToolResult:
    cleaned = (query or "").strip()
    if not cleaned:
        return ToolResult(False, "query is required")
    limit = _clamp_int(max_results, default=DEFAULT_SEARCH_RESULTS, high=MAX_SEARCH_RESULTS)
    results: list[dict[str, str]] = []
    errors: list[str] = []
    try:
        results.extend(_instant_answers(cleaned))
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    try:
        results.extend(_html_search(cleaned))
    except (httpx.HTTPError, ValueError) as exc:
        errors.append(str(exc))
    results = _dedupe_results(results)[:limit]
    if not results:
        detail = f" ({'; '.join(errors)})" if errors else ""
        return ToolResult(
            False,
            f"No web results. Try a more specific query or fetch_url on a known docs URL.{detail}",
        )
    return ToolResult(True, _format_search_results(results))


def fetch_url(url: str, max_chars: int = DEFAULT_FETCH_CHARS) -> ToolResult:
    target = _rewrite_github_blob((url or "").strip())
    if not target:
        return ToolResult(False, "url is required")
    limit = _clamp_int(max_chars, default=DEFAULT_FETCH_CHARS, high=MAX_FETCH_CHARS)
    try:
        final_url, content_type, body = _get_public_text(target)
    except ValueError as exc:
        return ToolResult(False, str(exc))
    except httpx.HTTPError as exc:
        return ToolResult(False, f"fetch_url failed: {exc}")
    kind = (content_type or "").split(";", 1)[0].strip().lower()
    if _is_binary_type(kind):
        return ToolResult(False, f"refusing non-text content: {kind or 'unknown'}")
    if "json" in kind or final_url.endswith(".json"):
        text = _pretty_json(body)
    elif "html" in kind or _looks_like_html(body):
        text = html_to_text(body)
    else:
        text = body.replace("\r\n", "\n")
    text = text.strip()
    if not text:
        return ToolResult(False, f"no readable text at {final_url}")
    truncated = ""
    if len(text) > limit:
        text = text[:limit].rstrip()
        truncated = f"\n\n... truncated to {limit} characters"
    return ToolResult(True, f"URL: {final_url}\n\n{text}{truncated}")


def check_public_url(url: str) -> str | None:
    """Return an error message when the URL is not a public http(s) address."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return "only http and https URLs are allowed"
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return "URL host is required"
    if host in _BLOCKED_HOSTS or host.endswith(".localhost") or host.endswith(".local"):
        return "refusing local or private URL"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if not address.is_global:
            return "refusing local or private URL"
        return None
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return f"could not resolve host: {host}"
    for info in infos:
        raw = info[4][0]
        try:
            resolved = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if not resolved.is_global:
            return "refusing local or private URL"
    return None


def html_to_text(markup: str) -> str:
    parser = _HTMLText()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 - broken HTML still yields whatever we parsed
        pass
    return parser.text()


def _instant_answers(query: str) -> list[dict[str, str]]:
    response = httpx.get(
        INSTANT_ENDPOINT,
        params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        headers=HEADERS,
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return []
    results: list[dict[str, str]] = []
    abstract = str(payload.get("AbstractText") or payload.get("Abstract") or "").strip()
    abstract_url = str(payload.get("AbstractURL") or "").strip()
    heading = str(payload.get("Heading") or "").strip()
    source = str(payload.get("AbstractSource") or "").strip()
    if abstract_url and (abstract or heading):
        title = heading or abstract_url
        if source:
            title = f"{title} ({source})"
        results.append({"title": title, "url": abstract_url, "snippet": abstract})
    for topic in payload.get("RelatedTopics") or []:
        if not isinstance(topic, dict):
            continue
        first_url = str(topic.get("FirstURL") or "").strip()
        text = str(topic.get("Text") or "").strip()
        if first_url and text:
            results.append({"title": text.split(" - ", 1)[0], "url": first_url, "snippet": text})
        if len(results) >= MAX_SEARCH_RESULTS:
            break
    return results


def _html_search(query: str) -> list[dict[str, str]]:
    response = httpx.post(
        SEARCH_ENDPOINT,
        data={"q": query},
        headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    results = parse_search_html(response.text)
    if results:
        return results
    response = httpx.get(
        SEARCH_ENDPOINT,
        params={"q": query},
        headers=HEADERS,
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    return parse_search_html(response.text)


def parse_search_html(markup: str) -> list[dict[str, str]]:
    parser = _SearchHTML()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 - keep any results already collected
        pass
    return parser.results


def _get_public_text(url: str) -> tuple[str, str, str]:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        error = check_public_url(current)
        if error:
            raise ValueError(error)
        response = httpx.get(
            current,
            headers=HEADERS,
            timeout=FETCH_TIMEOUT,
            follow_redirects=False,
        )
        if response.is_redirect:
            location = (response.headers.get("location") or "").strip()
            if not location:
                raise ValueError("redirect missing Location")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        body = response.content[:MAX_FETCH_BYTES]
        encoding = response.encoding or "utf-8"
        return str(response.url), content_type, body.decode(encoding, errors="replace")
    raise ValueError("too many redirects")


def _rewrite_github_blob(url: str) -> str:
    match = _GITHUB_BLOB_RE.match(url)
    if match is None:
        return url
    owner = match.group("owner")
    repo = match.group("repo")
    ref = match.group("ref")
    path = match.group("path")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"


def _unwrap_ddg(href: str) -> str:
    raw = (href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = f"https:{raw}"
    parsed = urlparse(raw)
    uddg = parse_qs(parsed.query).get("uddg")
    if uddg:
        return unquote(uddg[0])
    return raw


def _format_search_results(results: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        title = item.get("title") or item.get("url") or "Untitled"
        lines.append(f"{index}. {title}")
        if item.get("url"):
            lines.append(f"   URL: {item['url']}")
        snippet = (item.get("snippet") or "").strip()
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def _dedupe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in results:
        url = (item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique


def _pretty_json(body: str) -> str:
    try:
        return json.dumps(json.loads(body), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return body


def _looks_like_html(body: str) -> bool:
    start = body.lstrip()[:200].lower()
    return start.startswith("<!doctype html") or start.startswith("<html") or "<body" in start


def _is_binary_type(kind: str) -> bool:
    if not kind:
        return False
    if kind.startswith(("image/", "audio/", "video/", "font/")):
        return True
    return kind in {"application/octet-stream", "application/pdf", "application/zip"}


def _clamp_int(value: object, *, default: int, high: int) -> int:
    try:
        number = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, min(number, high))


class _SearchHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._href = ""
        self._title: list[str] = []
        self._snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = _class_set(attrs)
        href = _attr(attrs, "href")
        if tag == "a" and "result__a" in classes:
            self._flush_current()
            self._in_title = True
            self._href = _unwrap_ddg(href)
            self._title = []
            self._snippet = []
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            self._flush_current()
        elif self._in_snippet and tag in {"a", "td", "div"}:
            self._in_snippet = False
            self._attach_snippet()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif self._in_snippet:
            self._snippet.append(data)

    def close(self) -> None:
        self._flush_current()
        super().close()

    def _flush_current(self) -> None:
        title = html.unescape("".join(self._title)).strip()
        url = self._href.strip()
        if title and url:
            snippet = html.unescape("".join(self._snippet)).strip()
            self.results.append({"title": title, "url": url, "snippet": snippet})
        self._href = ""
        self._title = []
        self._snippet = []
        self._in_title = False
        self._in_snippet = False

    def _attach_snippet(self) -> None:
        snippet = html.unescape("".join(self._snippet)).strip()
        if snippet and self.results and not self.results[-1].get("snippet"):
            self.results[-1]["snippet"] = snippet
        self._snippet = []
        self._in_snippet = False


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._main: list[str] = []
        self._body: list[str] = []
        self._title: list[str] = []
        self._in_main = 0
        self._in_body = 0
        self._in_title = 0
        self._skip = 0
        self._saw_body = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        classes_role = _attr(attrs, "role").lower()
        if name in {"main", "article"} or classes_role == "main":
            self._in_main += 1
        if name == "body":
            self._saw_body = True
            self._in_body += 1
        if name == "title":
            self._in_title += 1
        if name == "br":
            self._emit("\n")
        elif name in _HEADING_TAGS:
            self._emit(f"\n{_HEADING_TAGS[name]} ")
        elif name == "li":
            self._emit("\n- ")
        elif name in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if name in {"main", "article"}:
            self._in_main = max(0, self._in_main - 1)
        if name == "body":
            self._in_body = max(0, self._in_body - 1)
        if name == "title":
            self._in_title = max(0, self._in_title - 1)
        if name in _HEADING_TAGS or name in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self._skip or not data:
            return
        if self._in_title:
            self._title.append(data)
            return
        self._emit(data)

    def text(self) -> str:
        parts = self._main or self._body
        body = _collapse_space("".join(parts))
        title = _collapse_space("".join(self._title))
        if title and title.lower() not in body.lower():
            return f"{title}\n\n{body}".strip()
        return body

    def _emit(self, data: str) -> None:
        if self._in_main:
            self._main.append(data)
        elif self._in_body or not self._saw_body:
            self._body.append(data)


def _collapse_space(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").splitlines()]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank and collapsed:
                collapsed.append("")
            blank = True
            continue
        blank = False
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def _class_set(attrs: list[tuple[str, str | None]]) -> set[str]:
    return set((_attr(attrs, "class") or "").split())


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
    for key, value in attrs:
        if key == name:
            return value or ""
    return ""
