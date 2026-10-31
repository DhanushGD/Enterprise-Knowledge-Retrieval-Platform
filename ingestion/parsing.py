"""Step 5: raw bytes -> structured text.

Every parser returns the same shape: a title plus a list of sections, each with its
heading path (e.g. ["Runner", "Install", "Linux"]). Keeping that structure is what lets
the chunker split on meaning instead of on character counts.

Markdown, HTML, DOCX and TXT use only the standard library. PDF needs `pypdf`.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET


class ParseError(Exception):
    """The document could not be turned into usable text."""


class EmptyDocumentError(ParseError):
    """The file parsed fine but holds no body text (e.g. a Hugo section index page).
    That is a normal outcome, not a failure: the pipeline skips it quietly."""


@dataclass
class Section:
    heading_path: list[str]
    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    title: str
    sections: list[Section]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "ParsedDocument":
        data = json.loads(raw)
        return cls(data["title"], [Section(**s) for s in data["sections"]])


# ---------------------------------------------------------------- shared helpers

def _decode(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _tidy(text: str, strip_lines: bool = False) -> str:
    lines = text.split("\n")
    lines = [ln.strip() for ln in lines] if strip_lines else [ln.rstrip() for ln in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _title_from_path(path: str) -> str:
    stem = PurePosixPath(path).stem
    return re.sub(r"[-_]+", " ", stem).strip().title() or "Untitled"


class _SectionBuilder:
    """Collects body text under the current heading and emits a Section per heading."""

    def __init__(self, strip_lines: bool = False):
        self.stack: list[tuple[int, str]] = []
        self.sections: list[Section] = []
        self.first_h1: str | None = None
        self._buf: list[str] = []
        self._strip_lines = strip_lines

    def heading(self, level: int, text: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        self._flush()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        if level == 1 and self.first_h1 is None:
            self.first_h1 = text

    def add(self, text: str) -> None:
        self._buf.append(text)

    def _flush(self) -> None:
        body = _tidy("\n".join(self._buf), self._strip_lines)
        if body:
            self.sections.append(Section([t for _, t in self.stack], body))
        self._buf = []

    def finish(self) -> list[Section]:
        self._flush()
        return self.sections


# -------------------------------------------------------------------- markdown

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_FM_TITLE = re.compile(r"^title:[ \t]*[\"']?(.+?)[\"']?[ \t]*$", re.M)
_TOML_FRONT = re.compile(r"\A\+\+\+[ \t]*\n(.*?)\n\+\+\+[ \t]*(?:\n|\Z)", re.S)
_TOML_TITLE = re.compile(r"^title[ \t]*=[ \t]*[\"'](.+?)[\"'][ \t]*$", re.M)
_HEADING_ATTR = re.compile(r"[ \t]*\{[#:.][^}]*\}[ \t]*$")   # "## Title {#anchor}"
_SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)   # Hugo: {{< alert >}} {{% note %}}
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_FENCE = re.compile(r"^\s*(```+|~~~+)")
_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LAYOUT_TAG = re.compile(r"</?(?:br|p|div|span|details|summary|center|font|sup|sub)\b[^>]*>", re.I)


def _clean_inline(line: str) -> str:
    line = _IMAGE.sub(r"\1", line)
    line = _LINK.sub(r"\1", line)
    return _LAYOUT_TAG.sub("", line)


def _parse_markdown(data: bytes, path: str) -> ParsedDocument:
    text = _decode(data)
    fm_title = None
    front = _FRONT_MATTER.match(text)
    toml = None if front else _TOML_FRONT.match(text)
    if front:
        m = _FM_TITLE.search(front.group(1))
        fm_title = m.group(1).strip() if m else None
        text = text[front.end():]
    elif toml:
        m = _TOML_TITLE.search(toml.group(1))
        fm_title = m.group(1).strip() if m else None
        text = text[toml.end():]
    text = _HTML_COMMENT.sub("", _SHORTCODE.sub("", text))

    builder = _SectionBuilder()
    in_fence, marker = False, ""
    for line in text.split("\n"):
        fence = _FENCE.match(line)
        if fence:
            if not in_fence:
                in_fence, marker = True, fence.group(1)[0]
            elif fence.group(1)[0] == marker:
                in_fence = False
            builder.add(line)
            continue
        if in_fence:
            builder.add(line)
            continue
        heading = _MD_HEADING.match(line)
        if heading:
            builder.heading(
                len(heading.group(1)), _HEADING_ATTR.sub("", _clean_inline(heading.group(2)))
            )
        else:
            builder.add(_clean_inline(line))

    title = fm_title or builder.first_h1 or _title_from_path(path)
    return ParsedDocument(title, builder.finish())


# ------------------------------------------------------------------------ html

class _HTMLExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "svg", "template"}
    BLOCK = {"p", "div", "section", "article", "br", "ul", "ol", "pre", "blockquote",
             "table", "tr", "hr", "dl", "dt", "dd", "main", "figure"}
    HEADINGS = {f"h{i}" for i in range(1, 7)}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.builder = _SectionBuilder(strip_lines=True)
        self.title: str | None = None
        self._skip = 0
        self._in_pre = 0
        self._in_title = False
        self._heading_level = 0
        self._heading_buf: list[str] = []
        self._pieces: list[str] = []

    def _flush_text(self) -> None:
        text = "".join(self._pieces)
        self._pieces = []
        text = re.sub(r"[ \t]*\|[ \t]*\n", "\n", text)   # drop trailing table separators
        if text.strip():
            self.builder.add(text)

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in self.HEADINGS:
            self._flush_text()
            self._heading_level, self._heading_buf = int(tag[1]), []
        elif tag == "title":
            self._in_title = True
        elif tag == "li":
            self._pieces.append("\n- ")
        elif tag == "pre":
            self._in_pre += 1
            self._pieces.append("\n")
        elif tag in self.BLOCK:
            self._pieces.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in self.HEADINGS and self._heading_level:
            heading = "".join(self._heading_buf)
            self.builder.heading(self._heading_level, heading)
            self._heading_level = 0
        elif tag == "title":
            self._in_title = False
        elif tag == "pre":
            self._in_pre = max(0, self._in_pre - 1)
            self._pieces.append("\n")
        elif tag in ("td", "th"):
            self._pieces.append(" | ")
        elif tag in self.BLOCK or tag == "li":
            self._pieces.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title = (self.title or "") + data.strip()
            return
        if not self._in_pre:
            data = re.sub(r"\s+", " ", data)
        if self._heading_level:
            self._heading_buf.append(data)
        else:
            self._pieces.append(data)

    def result(self) -> list[Section]:
        self._flush_text()
        return self.builder.finish()


def _parse_html(data: bytes, path: str) -> ParsedDocument:
    extractor = _HTMLExtractor()
    extractor.feed(_decode(data))
    extractor.close()
    sections = extractor.result()
    title = extractor.builder.first_h1 or extractor.title or _title_from_path(path)
    return ParsedDocument(title, sections)


# ------------------------------------------------------------------------ docx

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_STYLE = re.compile(r"^heading\s*([1-9])$", re.I)


def _docx_paragraph_text(p: ET.Element) -> str:
    parts = []
    for el in p.iter():
        if el.tag == f"{_W}t":
            parts.append(el.text or "")
        elif el.tag == f"{_W}tab":
            parts.append(" ")
        elif el.tag == f"{_W}br":
            parts.append("\n")
    return "".join(parts)


def _parse_docx(data: bytes, path: str) -> ParsedDocument:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        root = ET.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ParseError(f"not a valid .docx file: {exc}") from exc

    body = root.find(f"{_W}body")
    builder = _SectionBuilder()
    doc_title = None
    for child in (body if body is not None else []):
        if child.tag == f"{_W}p":
            text = _docx_paragraph_text(child).strip()
            if not text:
                continue
            ppr = child.find(f"{_W}pPr")
            style_el = ppr.find(f"{_W}pStyle") if ppr is not None else None
            style = style_el.get(f"{_W}val", "") if style_el is not None else ""
            match = _HEADING_STYLE.match(style.replace("_", " "))
            if style.lower() == "title":
                doc_title = doc_title or text
                builder.heading(0, text)   # Title sits above Heading 1, so H1s nest under it
            elif match:
                builder.heading(int(match.group(1)), text)
            elif ppr is not None and ppr.find(f"{_W}numPr") is not None:
                builder.add(f"- {text}")
            else:
                builder.add(text)
        elif child.tag == f"{_W}tbl":
            for row in child.iter(f"{_W}tr"):
                cells = [
                    " ".join(_docx_paragraph_text(p).strip() for p in cell.iter(f"{_W}p")).strip()
                    for cell in row.findall(f"{_W}tc")
                ]
                if any(cells):
                    builder.add(" | ".join(cells))
            builder.add("")

    title = doc_title or builder.first_h1 or _title_from_path(path)
    return ParsedDocument(title, builder.finish())


# ------------------------------------------------------------------------- pdf

def _parse_pdf(data: bytes, path: str) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ParseError("PDF support needs pypdf: pip install pypdf") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = list(reader.pages)
    except Exception as exc:  # pypdf raises many different error types
        raise ParseError(f"could not open PDF: {exc}") from exc

    sections = []
    for number, page in enumerate(pages, start=1):
        try:
            text = _tidy(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            sections.append(Section([], text, page=number))
    if not sections:
        raise ParseError("PDF has no extractable text (scanned? it needs OCR)")

    meta_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    return ParsedDocument(meta_title or _title_from_path(path), sections)


# ------------------------------------------------------------------------- txt

def _parse_text(data: bytes, path: str) -> ParsedDocument:
    text = _tidy(_decode(data))
    sections = [Section([], text)] if text else []
    return ParsedDocument(_title_from_path(path), sections)


# -------------------------------------------------------------------- dispatch

_PARSERS = {
    ".md": _parse_markdown, ".mdx": _parse_markdown,
    ".html": _parse_html, ".htm": _parse_html,
    ".docx": _parse_docx, ".pdf": _parse_pdf, ".txt": _parse_text,
}


def parse_document(data: bytes, path: str) -> ParsedDocument:
    parser = _PARSERS.get(PurePosixPath(path).suffix.lower())
    if parser is None:
        raise ParseError(f"unsupported file type: {path}")
    parsed = parser(data, path)
    if not parsed.sections:
        raise EmptyDocumentError("document has no text content")
    return parsed
