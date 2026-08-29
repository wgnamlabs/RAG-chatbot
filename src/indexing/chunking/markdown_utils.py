"""Markdown structure utilities that preserve source whitespace.

Why this exists
---------------
LangChain's ``MarkdownHeaderTextSplitter`` normalizes/strips leading whitespace
from ordinary lines. That is dangerous for this corpus because indentation is
semantic Markdown structure, e.g.::

    - Điều kiện
      - Tiêu chí 1
      - Tiêu chí 2

Flattening the two child bullets to column 0 weakens the parent/child relation.
This module performs the small amount of header parsing we need (H1-H6) while
leaving every non-heading line byte-for-byte unchanged.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")


@dataclass
class MarkdownSection:
    page_content: str
    metadata: dict


def split_markdown_sections(
    text: str,
    headers_to_split_on: Sequence[Tuple[str, str]],
) -> List[MarkdownSection]:
    """Split Markdown on configured headings without touching list indentation.

    The behavior intentionally mirrors the subset of
    ``MarkdownHeaderTextSplitter(strip_headers=False)`` needed by this project:

    - configured heading lines stay inside ``page_content``;
    - active heading values are exposed as ``Header 1`` ... ``Header 6`` metadata;
    - when a heading at level N appears, metadata at N and all deeper levels is
      replaced/cleared;
    - blank lines and indentation inside the section are preserved.
    """

    marker_to_name: Dict[str, str] = dict(headers_to_split_on)
    configured_levels = {
        len(marker): (marker, name)
        for marker, name in headers_to_split_on
        if marker and set(marker) == {"#"} and 1 <= len(marker) <= 6
    }

    sections: List[MarkdownSection] = []
    active_meta: Dict[str, str] = {}
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        # Only remove boundary newlines. Never call .strip(), because it would
        # destroy indentation if a section/chunk begins with a nested list item.
        content = "\n".join(buffer).strip("\n")
        if content.strip():
            sections.append(MarkdownSection(content, active_meta.copy()))
        buffer = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        marker = match.group(1) if match else None

        if match and marker in marker_to_name:
            flush()

            level = len(marker)
            # New heading at level N invalidates the previous heading at N and
            # every deeper heading.
            for configured_level, (_, meta_name) in configured_levels.items():
                if configured_level >= level:
                    active_meta.pop(meta_name, None)

            active_meta[marker_to_name[marker]] = match.group(2).strip()
            buffer.append(line)
            continue

        buffer.append(line)

    flush()
    return sections
