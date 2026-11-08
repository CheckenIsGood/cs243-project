"""Inline dominance.py's JSON into dominance_page.html. Usage: build_dominance_page.py data.json out.html [as_of]"""
import json, sys, time
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
d["generated"] = sys.argv[3] if len(sys.argv) > 3 else time.strftime("%Y-%m-%d %H:%M")
page = (Path(__file__).parent / "dominance_page.html").read_text()
Path(sys.argv[2]).write_text(page.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
