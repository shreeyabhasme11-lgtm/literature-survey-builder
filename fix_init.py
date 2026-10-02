"""
Run this once from the project root to create all missing __init__.py files.
Usage: python fix_init.py
"""
from pathlib import Path

ROOT = Path(__file__).parent

folders = [
    "app",
    "app/ui",
    "app/ingestion",
    "app/extraction",
    "app/metadata",
    "app/representation",
    "app/vectorstore",
    "app/ranking",
    "app/topics",
    "app/sections",
    "app/coherence",
    "app/composition",
    "app/explainability",
    "app/export",
    "app/utils",
]

for folder in folders:
    init_file = ROOT / folder / "__init__.py"
    init_file.parent.mkdir(parents=True, exist_ok=True)
    if not init_file.exists():
        init_file.write_text("# auto-generated\n")
        print(f"  Created: {init_file}")
    else:
        print(f"  Exists:  {init_file}")

print("\n✅ All __init__.py files are in place.")
print("Now run: streamlit run app/main.py")