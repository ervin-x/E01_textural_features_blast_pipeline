#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
UL_RE = re.compile(r"^-\s+(.*)$")
OL_RE = re.compile(r"^\d+\.\s+(.*)$")
TABLE_RULE_RE = re.compile(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")
CODE_RE = re.compile(r"`([^`]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
STRONG_RE = re.compile(r"\*\*([^*]+)\*\*")
EM_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def to_href(target: str) -> str:
    candidate = Path(target)
    if candidate.is_absolute() and candidate.exists():
        return candidate.as_uri()
    return html.escape(target, quote=True)


def to_image_src(target: str) -> str:
    candidate = Path(target)
    if candidate.is_absolute() and candidate.exists():
        mime_type, _ = mimetypes.guess_type(candidate.name)
        if mime_type:
            encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
    return html.escape(target, quote=True)


def render_inline(text: str) -> str:
    placeholders: list[str] = []

    def stash(fragment: str) -> str:
        placeholders.append(fragment)
        return f"@@PLACEHOLDER_{len(placeholders) - 1}@@"

    escaped = html.escape(text)

    def code_replace(match: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(match.group(1))}</code>")

    escaped = CODE_RE.sub(code_replace, escaped)

    def link_replace(match: re.Match[str]) -> str:
        label = render_inline(match.group(1))
        href = to_href(match.group(2))
        return stash(f'<a href="{href}">{label}</a>')

    escaped = LINK_RE.sub(link_replace, escaped)
    escaped = STRONG_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = EM_RE.sub(r"<em>\1</em>", escaped)

    for index, fragment in enumerate(placeholders):
        escaped = escaped.replace(f"@@PLACEHOLDER_{index}@@", fragment)
    return escaped


def split_table_row(line: str) -> list[str]:
    raw_cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in raw_cells]


def cell_alignment(spec: str) -> str:
    stripped = spec.strip()
    left = stripped.startswith(":")
    right = stripped.endswith(":")
    if left and right:
        return "center"
    if right:
        return "right"
    return "left"


def render_table(table_lines: list[str]) -> str:
    rows = [split_table_row(line) for line in table_lines]
    alignments: list[str] = []
    body_rows = rows[1:]
    if len(rows) > 1 and TABLE_RULE_RE.match(table_lines[1].strip()):
        alignments = [cell_alignment(spec) for spec in rows[1]]
        body_rows = rows[2:]

    header = rows[0]
    if not alignments:
        alignments = ["left"] * len(header)

    parts = ["<table>", "<thead>", "<tr>"]
    for cell, align in zip(header, alignments):
        parts.append(f'<th style="text-align: {align};">{render_inline(cell)}</th>')
    parts.extend(["</tr>", "</thead>", "<tbody>"])

    for row in body_rows:
        parts.append("<tr>")
        for idx, cell in enumerate(row):
            align = alignments[idx] if idx < len(alignments) else "left"
            parts.append(f'<td style="text-align: {align};">{render_inline(cell)}</td>')
        parts.append("</tr>")

    parts.extend(["</tbody>", "</table>"])
    return "\n".join(parts)


def markdown_to_html(markdown_text: str, title: str) -> str:
    lines = markdown_text.splitlines()
    body_parts: list[str] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []
    list_tag: str | None = None
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines)
        body_parts.append(f"<p>{render_inline(paragraph)}</p>")
        paragraph_lines.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if not list_items or not list_tag:
            return
        body_parts.append(f"<{list_tag}>")
        for item in list_items:
            body_parts.append(f"<li>{render_inline(item)}</li>")
        body_parts.append(f"</{list_tag}>")
        list_items.clear()
        list_tag = None

    def flush_table() -> None:
        if not table_lines:
            return
        body_parts.append(render_table(table_lines))
        table_lines.clear()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            flush_list()
            table_lines.append(stripped)
            continue

        flush_table()

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            flush_paragraph()
            flush_list()
            alt_text, image_target = image_match.groups()
            src = to_image_src(image_target)
            caption = render_inline(alt_text)
            body_parts.append(
                "\n".join(
                    [
                        '<figure class="report-figure">',
                        f'<img src="{src}" alt="{html.escape(alt_text, quote=True)}">',
                        f"<figcaption>{caption}</figcaption>",
                        "</figure>",
                    ]
                )
            )
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group(1))
            heading_text = render_inline(heading_match.group(2))
            body_parts.append(f"<h{level}>{heading_text}</h{level}>")
            continue

        unordered_match = UL_RE.match(stripped)
        if unordered_match:
            flush_paragraph()
            if list_tag not in (None, "ul"):
                flush_list()
            list_tag = "ul"
            list_items.append(unordered_match.group(1))
            continue

        ordered_match = OL_RE.match(stripped)
        if ordered_match:
            flush_paragraph()
            if list_tag not in (None, "ol"):
                flush_list()
            list_tag = "ol"
            list_items.append(ordered_match.group(0).split(". ", 1)[1])
            continue

        if list_items:
            flush_list()
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_list()
    flush_table()

    css = """
    @page {
        margin: 22mm 18mm 20mm 22mm;
    }
    body {
        font-family: "Times New Roman", serif;
        font-size: 12pt;
        line-height: 1.45;
        color: #111111;
        max-width: 175mm;
        margin: 0 auto;
    }
    h1, h2, h3, h4 {
        font-weight: 700;
        page-break-after: avoid;
        margin-top: 1.1em;
        margin-bottom: 0.45em;
    }
    h1 {
        text-align: center;
        font-size: 18pt;
        margin-top: 0.2em;
    }
    h2 {
        font-size: 15pt;
    }
    h3 {
        font-size: 13pt;
    }
    p {
        margin: 0 0 0.65em 0;
        text-align: justify;
    }
    ul, ol {
        margin: 0.2em 0 0.8em 1.5em;
    }
    li {
        margin: 0.2em 0;
    }
    code {
        font-family: "Courier New", monospace;
        font-size: 10.5pt;
        background: #f2f2f2;
        padding: 0.05em 0.25em;
        border-radius: 2px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.75em 0 1.0em 0;
        font-size: 10.5pt;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #444444;
        padding: 5px 7px;
        vertical-align: top;
    }
    th {
        background: #e9edf2;
    }
    .report-figure {
        margin: 1em auto 1.2em auto;
        text-align: center;
        page-break-inside: avoid;
    }
    .report-figure img {
        max-width: 100%;
        height: auto;
        border: 1px solid #b5b5b5;
    }
    .report-figure figcaption {
        margin-top: 0.45em;
        font-size: 10.5pt;
        font-style: italic;
    }
    a {
        color: #0b57d0;
        text-decoration: none;
    }
    """

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="ru">',
            "<head>",
            '  <meta charset="utf-8">',
            f"  <title>{html.escape(title)}</title>",
            "  <style>",
            css,
            "  </style>",
            "</head>",
            "<body>",
            *body_parts,
            "</body>",
            "</html>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown report to standalone HTML.")
    parser.add_argument("input_markdown", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()

    markdown_text = args.input_markdown.read_text(encoding="utf-8")
    html_text = markdown_to_html(markdown_text, title=args.input_markdown.stem)
    args.output_html.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
