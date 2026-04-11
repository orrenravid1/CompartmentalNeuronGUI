from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "check_docs_vocabulary.py"
SPEC = importlib.util.spec_from_file_location("check_docs_vocabulary_script", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
check_docs_vocabulary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_docs_vocabulary
SPEC.loader.exec_module(check_docs_vocabulary)


def write_docs_repo(
    tmp_path: Path,
    *,
    markdown: str,
    rule: dict[str, object],
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    docs = repo / "docs"
    architecture = docs / "architecture"
    architecture.mkdir(parents=True)
    (docs / "guide.md").write_text(markdown, encoding="utf-8")
    config_path = architecture / "docs-vocabulary.json"
    config_path.write_text(json.dumps({"rules": [rule]}, indent=2) + "\n", encoding="utf-8")
    return repo, config_path


def test_repo_docs_vocabulary_has_no_error_level_findings():
    errors, warnings = check_docs_vocabulary.scan_docs_vocabulary(ROOT)
    assert errors == []
    assert isinstance(warnings, list)


def test_docs_vocabulary_ignores_code_and_inline_code(tmp_path: Path):
    repo, config_path = write_docs_repo(
        tmp_path,
        markdown="""---
title: Guide
---

`document`

```python
document = Scene()
```
""",
        rule={
            "name": "prefer-scene",
            "severity": "warning",
            "pattern": "document",
            "match_mode": "word",
            "paths": ["docs"],
            "ignore_code_blocks": True,
            "ignore_inline_code": True,
            "ignore_front_matter": True,
            "message": "Prefer scene.",
        },
    )

    errors, warnings = check_docs_vocabulary.scan_docs_vocabulary(repo, config_path)

    assert errors == []
    assert warnings == []


def test_docs_vocabulary_warning_can_be_escalated(tmp_path: Path):
    repo, config_path = write_docs_repo(
        tmp_path,
        markdown="""---
title: Guide
---

This document explains the app.
""",
        rule={
            "name": "prefer-scene",
            "severity": "warning",
            "pattern": "document",
            "match_mode": "word",
            "paths": ["docs"],
            "ignore_code_blocks": True,
            "ignore_inline_code": True,
            "ignore_front_matter": True,
            "message": "Prefer scene.",
        },
    )

    errors, warnings = check_docs_vocabulary.scan_docs_vocabulary(repo, config_path)

    assert errors == []
    assert warnings == ["docs/guide.md:5: warning [prefer-scene] Prefer scene."]


def test_docs_vocabulary_error_is_always_fatal(tmp_path: Path):
    retired_term = "Document" + "Ready"
    repo, config_path = write_docs_repo(
        tmp_path,
        markdown="""---
title: Guide
---

""" + retired_term + """ is stale here.
""",
        rule={
            "name": "retired-term",
            "severity": "error",
            "pattern": retired_term,
            "match_mode": "word",
            "paths": ["docs"],
            "ignore_code_blocks": True,
            "ignore_inline_code": True,
            "ignore_front_matter": True,
            "message": "Retired protocol term.",
        },
    )

    errors, warnings = check_docs_vocabulary.scan_docs_vocabulary(repo, config_path)

    assert warnings == []
    assert errors == ["docs/guide.md:5: error [retired-term] Retired protocol term."]
