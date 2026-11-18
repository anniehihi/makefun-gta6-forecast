"""Build a synthetic final-shaped case (minute-level history, streams, categories) from a practice case, to check that
main.py handles the final's shape and stays far inside the 8-second limit.

    python tools/fake_final.py path/to/p00.in out_dir/h00.in
"""

import json
import math
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import parse_time  # noqa: E402


def main():
    case = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    day_start = parse_time(case["day_start"])
    start = day_start - timedelta(hours=40)
    v1 = case["known"][0]["viewers"]
    history = []
    for minute in range(41 * 60):
        t = start + timedelta(minutes=minute)
        # A smooth ramp into the release with a rising last half hour of hour 1.
        level = v1 * math.exp(-3 + 3 * minute / (40 * 60)) if minute < 40 * 60 else v1 * (0.9 + 0.2 * (minute - 2400) / 60)
        history.append({"at": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "viewers": int(level), "channels": 100})
    case.update(
        id="h00",
        history=history,
        streams=[{"login": "x", "viewers": 1000, "language": "en", "started_at": case["day_start"]}],
        categories_hourly=[],
    )
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[2]).write_text(json.dumps(case), encoding="utf-8")


if __name__ == "__main__":
    main()
