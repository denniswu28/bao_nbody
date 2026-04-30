"""
codebase_overview.py
--------------------
Generate a markdown overview of every module in src/ for code review.
For each .py file, lists:
  - module docstring
  - top-level functions and classes (signature + first docstring line)
  - imports

Run from repo root:
    python scripts/codebase_overview.py            # print to stdout
    python scripts/codebase_overview.py -o OVERVIEW.md
"""
import argparse
import ast
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _first_doc_line(node):
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.strip().splitlines()[0]


def _format_signature(func: ast.FunctionDef) -> str:
    args = []
    a = func.args
    n_pos = len(a.args)
    n_def = len(a.defaults)
    for i, arg in enumerate(a.args):
        if i >= n_pos - n_def:
            d = a.defaults[i - (n_pos - n_def)]
            args.append(f"{arg.arg}={ast.unparse(d)}")
        else:
            args.append(arg.arg)
    if a.vararg:
        args.append(f"*{a.vararg.arg}")
    for i, arg in enumerate(a.kwonlyargs):
        d = a.kw_defaults[i]
        if d is None:
            args.append(arg.arg)
        else:
            args.append(f"{arg.arg}={ast.unparse(d)}")
    if a.kwarg:
        args.append(f"**{a.kwarg.arg}")
    return f"{func.name}({', '.join(args)})"


def summarize_module(path: Path) -> str:
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"### {path.name}\n\n**SYNTAX ERROR:** {e}\n"

    rel = path.relative_to(REPO_ROOT)
    lines = [f"### `{rel}` ({len(src.splitlines())} lines)\n"]

    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        lines.append(f"> {mod_doc.strip().splitlines()[0]}\n")

    imports = []
    funcs = []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.extend(f"{mod}.{a.name}" for a in node.names)
        elif isinstance(node, ast.FunctionDef):
            funcs.append(node)
        elif isinstance(node, ast.AsyncFunctionDef):
            funcs.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)

    if imports:
        lines.append(f"**Imports:** {', '.join(sorted(set(imports)))}\n")

    if classes:
        lines.append("**Classes:**\n")
        for c in classes:
            doc = _first_doc_line(c)
            lines.append(f"- `{c.name}` — {doc}" if doc else f"- `{c.name}`")
            for n in c.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = _format_signature(n)
                    d = _first_doc_line(n)
                    lines.append(f"    - `{sig}`" + (f" — {d}" if d else ""))
        lines.append("")

    public = [f for f in funcs if not f.name.startswith("_")]
    private = [f for f in funcs if f.name.startswith("_")]

    if public:
        lines.append("**Functions:**\n")
        for f in public:
            sig = _format_signature(f)
            d = _first_doc_line(f)
            lines.append(f"- `{sig}`" + (f" — {d}" if d else ""))
        lines.append("")

    if private:
        names = ", ".join(f"`{f.name}`" for f in private)
        lines.append(f"**Private helpers:** {names}\n")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", help="Write to this file instead of stdout")
    parser.add_argument("--src", default=str(SRC_DIR),
                        help="Directory to scan (default: src/)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    files = sorted(p for p in src_dir.glob("*.py") if not p.name.startswith("_"))

    out = []
    out.append("# Codebase Overview\n")
    out.append(f"_Auto-generated from `{src_dir.relative_to(REPO_ROOT)}/`._\n")
    out.append("## Files\n")
    for p in files:
        out.append(f"- [`{p.name}`](#{p.stem.replace('_', '-')})")
    out.append("\n## Modules\n")
    for p in files:
        out.append(summarize_module(p))

    text = "\n".join(out)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote overview to {args.output} ({len(text.splitlines())} lines)")
    else:
        print(text)


if __name__ == "__main__":
    main()
