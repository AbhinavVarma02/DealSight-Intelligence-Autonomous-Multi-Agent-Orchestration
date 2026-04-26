"""Convert agent log lines (with ANSI colour codes) into HTML for the UI."""

from __future__ import annotations

import html
import re

ANSI_RE = re.compile(r"\x1b\[(\d+)m")
COLOR_MAP = {
    "91": "#b42318",
    "92": "#137333",
    "93": "#8a6100",
    "94": "#0b66c3",
    "95": "#7b2cbf",
    "96": "#007c89",
    "97": "#20211f",
}


def reformat(log_line: str) -> str:
    escaped = html.escape(log_line)

    def repl(match: re.Match[str]) -> str:
        code = match.group(1)
        if code == "0":
            return "</span>"
        color = COLOR_MAP.get(code)
        if color:
            return f'<span style="color:{color}">'
        return ""

    return ANSI_RE.sub(repl, escaped)
