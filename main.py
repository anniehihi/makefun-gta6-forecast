"""make.fun gta6-twitch-ccu-forecast-01 entry: forecast launch-day Twitch viewers of Grand Theft Auto VI, hour by hour.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
The forecaster itself lives in model.py (an analog ensemble over the case's comparable launches).
Standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader runs `python3 -P`, which leaves this directory off sys.path: put it back so model.py imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)


def fallback(case: dict) -> dict:
    """The house persistence forecast, used only if the model fails, so a case is never invalid."""
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first else 1.0
    return {
        "hours": [
            {"hour": h, "q": [round(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)), 2) for z in Z]}
            for h in range(2, 25)
        ]
    }


def valid(forecast: dict) -> bool:
    rows = forecast.get("hours")
    if not isinstance(rows, list) or sorted(r.get("hour") for r in rows) != list(range(2, 25)):
        return False
    for r in rows:
        q = r.get("q")
        if not isinstance(q, list) or len(q) != 5:
            return False
        if any(not math.isfinite(x) or x <= 0 or x > 1e9 for x in q):
            return False
        if any(q[i] > q[i + 1] for i in range(4)):
            return False
    return True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = model.forecast_case(case)
            if not valid(forecast):
                raise ValueError("model produced an invalid forecast")
        except Exception as exc:  # never leave a case invalid
            print(f"{path.name}: falling back to persistence ({exc})", file=sys.stderr)
            forecast = fallback(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
