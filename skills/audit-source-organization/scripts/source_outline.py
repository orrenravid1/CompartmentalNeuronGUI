from __future__ import annotations

import argparse
import ast
from pathlib import Path


def node_length(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    start = getattr(node, "lineno", None)
    if not isinstance(end, int) or not isinstance(start, int):
        return 0
    return end - start + 1


def format_length(length: int) -> str:
    return f" [{length} lines]" if length else ""


def outline(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = [str(path)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(f"{node.lineno:4}: function {node.name}{format_length(node_length(node))}")
            continue
        if isinstance(node, ast.ClassDef):
            lines.append(f"{node.lineno:4}: class {node.name}{format_length(node_length(node))}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines.append(f"{child.lineno:4}:   method {child.name}{format_length(node_length(child))}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a names-only AST outline for Python source files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for index, path in enumerate(args.paths):
        if index:
            print()
        print(outline(path))


if __name__ == "__main__":
    main()
