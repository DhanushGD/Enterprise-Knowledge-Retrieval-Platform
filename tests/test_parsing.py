import io
import zipfile

import pytest

from ingestion.parsing import ParseError, ParsedDocument, parse_document


def last(section):
    return section.heading_path[-1] if section.heading_path else None


def paths(doc):
    return [s.heading_path for s in doc.sections]


# ---------------------------------------------------------------- markdown

MD = """---
title: "Install the Runner"
weight: 10
---

Intro text before any heading.

# Runner

Overview paragraph.

## Install

Steps here.

### Linux

Run the installer.

```bash
# this is a comment, NOT a heading
sudo ./install.sh
```

## Configure

See the [config guide](https://example.com/config) and ![diagram](img.png).

{{< alert type="note" >}}
Keep your token secret.
{{< /alert >}}

<!-- hidden comment -->
"""


def test_markdown_heading_paths_and_front_matter_title():
    doc = parse_document(MD.encode(), "runner/install.md")
    assert doc.title == "Install the Runner"
    assert paths(doc) == [
        [], ["Runner"], ["Runner", "Install"],
        ["Runner", "Install", "Linux"], ["Runner", "Configure"],
    ]


def test_markdown_code_fence_hash_is_not_a_heading():
    doc = parse_document(MD.encode(), "x.md")
    linux = next(s for s in doc.sections if last(s) == "Linux")
    assert "# this is a comment, NOT a heading" in linux.text
    assert "sudo ./install.sh" in linux.text


def test_markdown_links_images_shortcodes_comments_cleaned():
    doc = parse_document(MD.encode(), "x.md")
    cfg = next(s for s in doc.sections if last(s) == "Configure").text
    assert "config guide" in cfg and "https://example.com" not in cfg
    assert "diagram" in cfg and "img.png" not in cfg
    assert "{{" not in cfg and "Keep your token secret." in cfg
    assert "hidden comment" not in cfg


def test_markdown_title_falls_back_to_h1_then_filename():
    assert parse_document(b"# Real Title\n\nbody", "a.md").title == "Real Title"
    assert parse_document(b"just body text", "my-cool_doc.md").title == "My Cool Doc"


def test_markdown_heading_with_trailing_hash_char_kept():
    doc = parse_document(b"## Learning C#\n\ntext", "a.md")
    assert doc.sections[0].heading_path == ["Learning C#"]


def test_markdown_headings_without_body_produce_no_empty_sections():
    doc = parse_document(b"# A\n## B\ntext under B", "a.md")
    assert paths(doc) == [["A", "B"]]


def test_markdown_crlf_and_bom():
    doc = parse_document("﻿# T\r\n\r\nbody\r\n".encode("utf-8"), "a.md")
    assert doc.sections[0].text == "body"


# -------------------------------------------------------------------- html

HTML = """<html><head><title>Page Title</title><style>.x{}</style></head>
<body><nav>Menu Home About</nav>
<h1>Webhooks</h1><p>Intro   text.</p>
<h2>Retries</h2><ul><li>First try</li><li>Second try</li></ul>
<table><tr><th>Plan</th><th>SLA</th></tr><tr><td>Pro</td><td>8h</td></tr></table>
<script>var secret = 1;</script><footer>Copyright</footer></body></html>"""


def test_html_structure_and_noise_removal():
    doc = parse_document(HTML.encode(), "w.html")
    assert doc.title == "Webhooks"
    assert paths(doc) == [["Webhooks"], ["Webhooks", "Retries"]]
    full = " ".join(s.text for s in doc.sections)
    assert "Menu" not in full and "secret" not in full and "Copyright" not in full
    assert doc.sections[0].text == "Intro text."
    assert "- First try" in doc.sections[1].text and "- Second try" in doc.sections[1].text
    assert "Plan | SLA" in doc.sections[1].text and "Pro | 8h" in doc.sections[1].text


def test_html_title_tag_used_when_no_h1():
    doc = parse_document(b"<html><head><title>Only Title</title></head><body><p>hi</p></body></html>", "a.html")
    assert doc.title == "Only Title"


# -------------------------------------------------------------------- docx

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def make_docx() -> bytes:
    def p(text, style=None, bullet=False):
        ppr = ""
        if style:
            ppr += f'<w:pStyle w:val="{style}"/>'
        if bullet:
            ppr += '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
        ppr = f"<w:pPr>{ppr}</w:pPr>" if ppr else ""
        return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"

    table = (
        "<w:tbl><w:tr><w:tc>" + p("Plan") + "</w:tc><w:tc>" + p("SLA") + "</w:tc></w:tr>"
        "<w:tr><w:tc>" + p("Pro") + "</w:tc><w:tc>" + p("8h") + "</w:tc></w:tr></w:tbl>"
    )
    body = (
        p("Support Handbook", "Title") + p("Welcome text.")
        + p("Escalation", "Heading1") + p("Call the lead.") + p("Page them", bullet=True)
        + p("Targets", "Heading2") + table
    )
    xml = f'<?xml version="1.0"?><w:document {W}><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_docx_headings_lists_and_tables():
    doc = parse_document(make_docx(), "handbook.docx")
    assert doc.title == "Support Handbook"
    assert paths(doc) == [
        ["Support Handbook"], ["Support Handbook", "Escalation"],
        ["Support Handbook", "Escalation", "Targets"],
    ]
    escalation = next(s for s in doc.sections if last(s) == "Escalation")
    assert "Call the lead." in escalation.text and "- Page them" in escalation.text
    targets = next(s for s in doc.sections if last(s) == "Targets")
    assert "Plan | SLA" in targets.text and "Pro | 8h" in targets.text


def test_docx_corrupt_raises_parse_error():
    with pytest.raises(ParseError):
        parse_document(b"this is not a zip", "x.docx")


# --------------------------------------------------------------- txt / misc

def test_txt():
    doc = parse_document(b"line one\n\nline two", "notes.txt")
    assert doc.title == "Notes" and doc.sections[0].text == "line one\n\nline two"


def test_empty_and_unsupported_raise():
    with pytest.raises(ParseError):
        parse_document(b"   \n\n  ", "a.txt")
    with pytest.raises(ParseError):
        parse_document(b"# only a heading", "a.md")
    with pytest.raises(ParseError):
        parse_document(b"data", "a.xyz")


def test_invalid_pdf_raises_parse_error_whether_or_not_pypdf_is_installed():
    with pytest.raises(ParseError):
        parse_document(b"%PDF-garbage", "a.pdf")


def test_json_roundtrip():
    doc = parse_document(MD.encode(), "x.md")
    assert ParsedDocument.from_json(doc.to_json()) == doc


def test_toml_front_matter_title_and_removal():
    doc = parse_document(b'+++\ntitle = "Support Handbook"\nweight = 3\n+++\n\n## Intro\n\ntext', "_index.md")
    assert doc.title == "Support Handbook"
    assert "weight" not in doc.sections[0].text and "+++" not in doc.sections[0].text


def test_heading_anchors_are_stripped():
    doc = parse_document(b"# Guide {#guide}\n\n## Setup {#setup .class}\n\ntext", "a.md")
    assert doc.sections[0].heading_path == ["Guide", "Setup"]
    assert doc.title == "Guide"
