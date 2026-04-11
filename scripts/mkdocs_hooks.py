from __future__ import annotations

import re


FRONT_MATTER_PATTERN = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)


def on_page_markdown(markdown: str, **kwargs) -> str:
    del kwargs
    return FRONT_MATTER_PATTERN.sub("", markdown, count=1)
