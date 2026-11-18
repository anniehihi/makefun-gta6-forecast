"""Leave-one-launch-out backtest of model.py on the comparables of a practice case (there is no local grader).

Each comparable launch with a recorded release day becomes a pseudo-case: its hour-1 count and the 24 hours before it
are the input, the other comparables the analogs, its hours 2..24 the answer. The score is the contest's: the mean of
2 * max(tau * d, (tau - 1) * d), d = ln y - ln q, over every hour and level.

    python tools/backtest.py path/to/p00.in [--big] [key=value ...]

--big scores only launches whose hour 1 already had 20,000+ viewers (the regime GTA VI's release day is in).
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model  # noqa: E402

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)


def pinball(y: float, qs: list[float]) -> float:
    total = 0.0
    for tau, q in zip(model.LEVELS, qs):
        d = math.log(max(1.0, y)) - math.log(q)
        total += 2 * max(tau * d, (tau - 1) * d)
    return total / 5


def load_launches(case_path: str) -> list[model.Launch]:
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    launches = []
    for comp in case["comparables"]:
        launch = model.launch_from_comparable(comp)
        if launch is not None:
            launches.append(launch)
    return launches


def evaluate(launches, params=None, big=False, verbose=False):
    losses, base_losses = [], []
    per = []
    for i, target in enumerate(launches):
        if big and target.v1 < 20000:
            continue
        analogs = launches[:i] + launches[i + 1:]
        logq = model.forecast_log_quantiles(target, analogs, params)
        l, b = [], []
        for h, r in target.ratios.items():
            y = target.v1 * math.exp(r)
            l.append(pinball(y, [target.v1 * math.exp(x) for x in logq[h]]))
            b.append(pinball(y, [target.v1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]))
        losses += l
        base_losses += b
        per.append((target.name, sum(l) / len(l), sum(b) / len(b)))
    if verbose:
        for name, l, b in per:
            print(f"  {name[:32]:32s} model {l:.3f}  baseline {b:.3f}")
    return sum(losses) / len(losses), sum(base_losses) / len(base_losses)


def main():
    args = sys.argv[1:]
    sys.stdout.reconfigure(encoding="utf-8")
    path = args[0]
    big = "--big" in args
    params = {}
    for a in args[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            params[k] = (v == "True") if v in ("True", "False") else float(v)
    launches = load_launches(path)
    loss, base = evaluate(launches, params, big=big, verbose=True)
    print(f"model {loss:.4f}  baseline {base:.4f}  ({'big launches' if big else 'all launches'})")


if __name__ == "__main__":
    main()
