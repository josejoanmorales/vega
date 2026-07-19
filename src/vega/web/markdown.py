"""A ~40-line renderer for exactly the markdown subset Vega's briefings use
(h1-h3, `|`-tables, `**bold**`, `-`-lists, `---` rules, `_italic_`) — no CDN,
no JS libraries, deterministic. Anything outside this subset degrades to a
plain paragraph rather than crashing (a briefing is always readable, never
a stack trace on the page)."""

from __future__ import annotations

import html
import re

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!_)_([^_]+)_(?!_)")


def _inline(text: str) -> str:
    text = html.escape(text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    return _ITALIC.sub(r"<i>\1</i>", text)


def render_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif (
            line.startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) <= {"|", "-", " ", ":"}
        ):
            out.append("<table>")
            header = [c.strip() for c in line.strip("|").split("|")]
            out.append("<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</table>")
            continue
        elif line.startswith("- "):
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(out)
