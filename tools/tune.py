"""Coarse coordinate search of model.PARAMS on the leave-one-launch-out backtest (tools/backtest.py).

    python tools/tune.py path/to/p00.in

The objective is the mean of the big-launch loss (the regime of GTA VI's release day) and the all-launch loss, so
the parameters stay sane for a small target too. Grids are coarse on purpose: 37 launches do not support fine tuning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model  # noqa: E402
from backtest import evaluate, load_launches  # noqa: E402

GRID = {
    "bw_size": [0.3, 0.45, 0.6, 0.9, 1.3],
    "bw_gap": [0.2, 0.3, 0.45, 0.7, 1.0, 2.0],
    "bw_mom": [0.2, 0.3, 0.5, 0.8, 1.5, 5.0],
    "kern_base": [0.0, 0.03, 0.06, 0.1, 0.15],
    "kern_slope": [0.0, 0.03, 0.05, 0.08, 0.12],
    "ridge": [0.5, 1.0, 2.0, 4.0, 8.0, 30.0],
    "adj": [0.0, 0.5, 1.0],
    "w_floor": [0.0, 0.01, 0.02, 0.05, 0.1],
    "dst_shift": [False, True],
    "widen": [0.8, 0.9, 1.0, 1.1, 1.25],
}


def objective(launches, params):
    big, _ = evaluate(launches, params, big=True)
    every, _ = evaluate(launches, params, big=False)
    return (big + every) / 2, big, every


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    launches = load_launches(sys.argv[1])
    params = dict(model.PARAMS)
    best, big, every = objective(launches, params)
    print(f"start {best:.4f} (big {big:.4f}, all {every:.4f})")
    for _ in range(2):
        improved = False
        for key, values in GRID.items():
            for v in values:
                trial = dict(params, **{key: v})
                score, b, e = objective(launches, trial)
                if score < best - 1e-4:
                    best, params, improved = score, trial, True
                    print(f"  {key}={v}: {score:.4f} (big {b:.4f}, all {e:.4f})")
        if not improved:
            break
    print("best", best, params)


if __name__ == "__main__":
    main()
