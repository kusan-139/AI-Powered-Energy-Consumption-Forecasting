"""
generate_notebooks.py
=====================
Converts the .py notebook scripts into proper Jupyter .ipynb files.

Usage:
    python notebooks/generate_notebooks.py

Requirements:
    pip install nbformat nbconvert
"""

import sys
import nbformat
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
NB_DIR     = ROOT / "notebooks"

# Map: source .py → output .ipynb
NOTEBOOKS = [
    ("01_data_exploration.py",     "01_data_exploration.ipynb"),
    ("02_feature_engineering.py",  "02_feature_engineering.ipynb"),
    ("03_model_training.py",       "03_model_training.ipynb"),
    ("04_model_evaluation.py",     "04_model_evaluation.ipynb"),
    ("05_forecasting_dashboard.py","05_forecasting_dashboard.ipynb"),
]


def py_to_notebook(py_path: Path, ipynb_path: Path):
    """
    Convert a Python script to a Jupyter notebook.
    Lines starting with '# %%' or triple-quoted docstrings become Markdown cells.
    Everything else becomes a Code cell.
    """
    source = py_path.read_text(encoding="utf-8")
    lines  = source.splitlines(keepends=True)

    cells       = []
    buf         = []
    in_docstring= False
    doc_buf     = []

    def flush_code():
        code = "".join(buf).strip()
        if code:
            cells.append(nbformat.v4.new_code_cell(code))
        buf.clear()

    def flush_markdown():
        md = "".join(doc_buf).strip().strip('"""').strip()
        if md:
            cells.append(nbformat.v4.new_markdown_cell(md))
        doc_buf.clear()

    for line in lines:
        stripped = line.strip()

        # Detect docstring blocks (used as Markdown headings)
        if not in_docstring and stripped.startswith('"""'):
            flush_code()
            in_docstring = True
            doc_buf.append(line)
            if stripped.count('"""') >= 2 and len(stripped) > 6:
                in_docstring = False
                flush_markdown()
            continue

        if in_docstring:
            doc_buf.append(line)
            if '"""' in stripped:
                in_docstring = False
                flush_markdown()
            continue

        # Section separator → Markdown heading
        if stripped.startswith("# ══") or stripped.startswith("# ──") or stripped.startswith("# =="):
            continue  # Skip decorative lines

        if stripped.startswith("# ─") and len(stripped) > 30:
            continue

        # Cell split comment (# %%)
        if stripped.startswith("# %%"):
            flush_code()
            md_text = stripped[4:].strip()
            if md_text:
                cells.append(nbformat.v4.new_markdown_cell(f"## {md_text}"))
            continue

        buf.append(line)

    flush_code()

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language"    : "python",
            "name"        : "python3",
        },
        "language_info": {
            "name"   : "python",
            "version": "3.10.0",
        },
    }
    nbformat.write(nb, str(ipynb_path))
    print(f"  ✓  {ipynb_path.name}")


def main():
    print("\n[NOTEBOOKS] Converting .py scripts → .ipynb …\n")
    for src_name, dst_name in NOTEBOOKS:
        src = NB_DIR / src_name
        dst = NB_DIR / dst_name
        if not src.exists():
            print(f"  ✗  {src_name} not found — skipping")
            continue
        try:
            py_to_notebook(src, dst)
        except Exception as e:
            print(f"  ✗  {src_name} failed: {e}")

    print(f"\n[NOTEBOOKS] Done — open with:  jupyter notebook {NB_DIR}")
    print("  Or:  jupyter lab")


if __name__ == "__main__":
    main()
