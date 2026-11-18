"""Analog-ensemble forecaster of a launch day's hourly Twitch viewers.

The idea: every comparable launch in the case is a past "release day" whose curve, divided by its own hour-1 count,
tells how a launch day can unfold after its first hour. We put each comparable's log ratio curve
r_j(h) = ln(viewers_j(h) / viewers_j(1)) on the target's clock, weight the comparables by how much their situation at
the end of hour 1 resembles the target's, correct each analog for the remaining difference in those features with a
pooled linear adjustment, and read the five quantiles of each hour from a kernel mixture around the adjusted analogs.

Features of a launch at the end of hour 1 (computed the same way for the target and every comparable):
  size  log10 of hour 1's count: tiny categories are launches that have not unlocked yet and explode later;
  gap   ln(hour 1 / the highest hour of the 24 before it): a launch whose day-before audience was larger (early
        unlock in Asia/Oceania, US daytime hype) climbs back above hour 1 once the US wakes up;
  mom   ln(hour 1 / the hour before it): the short-term direction at the cut.

Pure standard library, deterministic (no randomness at all).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HOURS = range(2, 25)
# Final only (minute reads): the share of hour 1's closing trend carried into hour 2, and its fade per hour after.
TREND_CARRY = 0.6
TREND_DECAY = 0.6

# Tuned by leave-one-launch-out backtests over the comparables (tools/backtest.py).
PARAMS = {
    # Kernel bandwidths of the similarity weights, per feature.
    "bw_size": 0.45,
    "bw_gap": 1.0,
    "bw_mom": 1.5,
    # Spread of each analog's kernel in log units: base + slope * sqrt(h - 1).
    "kern_base": 0.15,
    "kern_slope": 0.03,
    # Ridge shrinkage of the pooled per-hour regression adjustment on the features.
    "ridge": 8.0,
    # Share of the regression adjustment applied (0 = plain analogs).
    "adj": 1.0,
    # Weight floor, so far analogs still count a little (robustness against an unusual target).
    "w_floor": 0.01,
    # Shift analogs between US daylight-saving and standard time so local-time patterns line up.
    "dst_shift": False,
    # Extra multiplicative widening of the mixture around its median.
    "widen": 1.0,
}


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _nth_sunday(year: int, month: int, n: int) -> datetime:
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    offset = (6 - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def us_dst(day: datetime) -> bool:
    """Whether US daylight-saving time is in effect on that UTC date (2007+ rules, good enough for every launch here)."""
    start = _nth_sunday(day.year, 3, 2)
    end = _nth_sunday(day.year, 11, 1)
    return start <= day < end


class Launch:
    """One launch day: its hour-1 count, the features at the cut and (for a comparable) its curve's log ratios."""

    def __init__(self, name: str, day_start: datetime, hourly: dict[datetime, float], v1: float):
        self.name = name
        self.day_start = day_start
        self.v1 = max(1.0, v1)
        before = [v for t, v in hourly.items() if day_start - timedelta(hours=24) <= t < day_start]
        prev = hourly.get(day_start - timedelta(hours=1))
        self.size = math.log10(self.v1)
        self.gap = math.log(self.v1 / max(1.0, max(before))) if before else 0.0
        self.mom = math.log(self.v1 / max(1.0, prev)) if prev else 0.0
        self.dst = us_dst(day_start)
        # Log ratios of hours 2..24 to hour 1 (None where the hour has no point).
        self.ratios: dict[int, float] = {}
        for h in range(2, 25):
            v = hourly.get(day_start + timedelta(hours=h - 1))
            if v is not None:
                self.ratios[h] = math.log(max(1.0, v) / self.v1)

    def features(self) -> tuple[float, float, float]:
        return (self.size, self.gap, self.mom)

    def ratio_on(self, h: int, shift: int) -> float | None:
        """The log ratio at the target's hour h, read `shift` hours earlier on this launch's clock, interpolated
        over a missing hour."""
        src = h - shift
        if src in self.ratios:
            return self.ratios[src]
        if src <= 1:
            return 0.0 if src == 1 else None
        lo, hi = self.ratios.get(src - 1), self.ratios.get(src + 1)
        if lo is not None and hi is not None:
            return (lo + hi) / 2
        return None


def launch_from_comparable(comp: dict) -> Launch | None:
    day_start = parse_time(comp["day_start"])
    hourly = {parse_time(r["from"]): float(r["viewers"]) for r in comp.get("hourly") or [] if r.get("viewers") is not None}
    v1 = hourly.get(day_start)
    if v1 is None or v1 <= 0:
        return None
    launch = Launch(comp.get("name", "?"), day_start, hourly, v1)
    if len(launch.ratios) < 18:
        return None
    return launch


def launch_from_case(case: dict) -> Launch:
    day_start = parse_time(case["day_start"])
    hourly = {parse_time(r["from"]): float(r["viewers"]) for r in case.get("history_hourly") or [] if r.get("viewers") is not None}
    first = next(k for k in case["known"] if k["hour"] == 1)
    v1 = float(first["viewers"])
    hourly[day_start] = v1
    return Launch(case.get("game", {}).get("name", "?"), day_start, hourly, v1)


# ---------------------------------------------------------------------------------------------------------------------
# The forecast


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _mixture_quantile(centers: list[float], weights: list[float], sd: float, tau: float) -> float:
    total = sum(weights)
    lo = min(centers) - 8 * sd
    hi = max(centers) + 8 * sd
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = sum(w * _normal_cdf((mid - c) / sd) for c, w in zip(centers, weights)) / total
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for k in range(col, n + 1):
                    m[r][k] -= f * m[col][k]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def forecast_log_quantiles(target: Launch, analogs: list[Launch], params: dict | None = None) -> dict[int, list[float]]:
    """The five quantiles of ln(y_h / v1) for each hour h = 2..24."""
    p = dict(PARAMS)
    if params:
        p.update(params)
    bws = (p["bw_size"], p["bw_gap"], p["bw_mom"])
    tf = target.features()

    weights = []
    for a in analogs:
        d2 = sum(((x - y) / bw) ** 2 for x, y, bw in zip(tf, a.features(), bws))
        weights.append(p["w_floor"] + math.exp(-0.5 * d2))

    out: dict[int, list[float]] = {}
    for h in HOURS:
        pts = []
        for a, w in zip(analogs, weights):
            shift = (int(a.dst) - int(target.dst)) if p["dst_shift"] else 0
            # An analog in daylight time lives one UTC hour earlier than a target in standard time: its hour h-1
            # pattern belongs at the target's hour h (shift = +1), and the other way around.
            r = a.ratio_on(h, shift)
            if r is None:
                r = a.ratio_on(h, 0)
            if r is not None:
                pts.append((a, w, r))
        if not pts:
            out[h] = [0.0] * 5
            continue

        # Pooled weighted ridge regression of r on the centred features, to adjust each analog toward the target.
        beta = [0.0, 0.0, 0.0]
        if p["adj"] > 0 and len(pts) >= 6:
            wsum = sum(w for _, w, _ in pts)
            mean_x = [sum(w * a.features()[k] for a, w, _ in pts) / wsum for k in range(3)]
            mean_r = sum(w * r for _, w, r in pts) / wsum
            xtx = [[0.0] * 3 for _ in range(3)]
            xty = [0.0] * 3
            for a, w, r in pts:
                x = [a.features()[k] - mean_x[k] for k in range(3)]
                for i in range(3):
                    xty[i] += w * x[i] * (r - mean_r)
                    for j in range(3):
                        xtx[i][j] += w * x[i] * x[j]
            for i in range(3):
                xtx[i][i] += p["ridge"] * (wsum / len(pts))
            beta = _solve(xtx, xty)

        centers, ws = [], []
        for a, w, r in pts:
            adj = sum(beta[k] * (tf[k] - a.features()[k]) for k in range(3)) * p["adj"]
            centers.append(r + adj)
            ws.append(w)
        sd = p["kern_base"] + p["kern_slope"] * math.sqrt(h - 1)
        qs = [_mixture_quantile(centers, ws, sd, tau) for tau in LEVELS]
        med = qs[2]
        qs = [med + (q - med) * p["widen"] for q in qs]
        out[h] = qs
    return out


def intra_hour_trend(case: dict, target: Launch) -> float:
    """ln(mean of hour 1's last 10 minute reads / hour 1's print), or 0 without minute-level reads.

    The final's history holds the recorder's minute reads, so where hour 1 ends is known more precisely than its
    mean; the practice's hourly points carry no such information (and give 0 here).
    """
    start = target.day_start
    end = start + timedelta(hours=1)
    reads = []
    for point in case.get("history") or []:
        t = parse_time(point["at"])
        if start <= t < end and point.get("viewers") is not None:
            reads.append((t, float(point["viewers"])))
    if len(reads) < 30:
        return 0.0
    reads.sort()
    tail = [v for _, v in reads[-10:]]
    trend = math.log(max(1.0, sum(tail) / len(tail)) / target.v1)
    return max(-0.5, min(0.5, trend))


def forecast_case(case: dict, params: dict | None = None) -> dict:
    """The contest's output object for one case."""
    target = launch_from_case(case)
    trend = intra_hour_trend(case, target)
    analogs = []
    for comp in case.get("comparables") or []:
        launch = launch_from_comparable(comp)
        if launch is not None:
            analogs.append(launch)
    rows = []
    hours = [row["hour"] for row in case["forecast"]]
    if analogs:
        logq = forecast_log_quantiles(target, analogs, params)
    else:
        logq = {}
    z = (-1.645, -0.674, 0.0, 0.674, 1.645)
    for h in hours:
        lq = logq.get(h) or [zz * 0.15 * math.sqrt(h - 1) for zz in z]
        # Damped carry-over of the end-of-hour-1 level: most of it in hour 2, fading over the next hours.
        shift = TREND_CARRY * trend * TREND_DECAY ** (h - 2)
        lq = [x + shift for x in lq]
        qs = []
        prev = 0.0
        for x in lq:
            v = min(1e9, max(0.01, target.v1 * math.exp(x)))
            v = max(v, prev)
            v = round(v, 2)
            qs.append(v)
            prev = v
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}
