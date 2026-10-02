import json
from pathlib import Path
from typing import List


def export_json(traces: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traces, indent=2, ensure_ascii=False))