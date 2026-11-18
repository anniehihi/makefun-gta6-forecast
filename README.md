# GTA VI launch-day Twitch viewers: analog-ensemble forecaster

Entry for the make.fun bounty *Forecast GTA VI's launch-day Twitch viewers hour by hour*
(`gta6-twitch-ccu-forecast-01`). Given the Twitch viewer count of the Grand Theft Auto VI category up to the end of
hour 1 of release day (2026-11-19 UTC), it writes five quantiles (5/25/50/75/95 %) for each of hours 2-24.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=path/to/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13 (3.11+ works), standard library only, no network, deterministic, ~0.2 s per case.

## Method (`model.py`)

Every comparable launch in the case is a past release day. Its curve, divided by its own hour-1 count, is a sample of
how the hours after hour 1 can unfold: `r_j(h) = ln(v_j(h) / v_j(1))`.

1. **Features at the cut**, computed the same way for the target and every comparable:
   `size` = log10 of hour 1 (tiny categories have not unlocked yet and explode later), `gap` = ln(hour 1 / highest
   hour of the previous 24) (an audience that was larger the day before comes back when the US wakes up), and `mom`
   = ln(hour 1 / the hour before).
2. **Similarity weights**: a Gaussian kernel on the features, plus a small floor so no analog is ever ignored.
3. **Clock alignment** (optional, `dst_shift`): analogs released under US daylight time can be shifted one hour so the US-East midnight unlock
   and the US-evening peak line up with a November release; the backtest preferred it off, so it is off.
4. **Regression adjustment**: per hour, a weighted ridge regression of `r` on the features moves each analog toward
   the target's features.
5. **Quantiles** of a Gaussian kernel mixture around the adjusted analogs (bandwidth grows with the horizon).
6. **Final-only nudge**: the final case carries minute reads, so the level at the end of hour 1 (last 10 minutes vs
   the hour mean) is carried, damped, into the first few hours.

`main.py` validates the output shape and falls back to the house persistence forecast if anything ever fails, so a
case is never invalid.

## Backtest

There is no local grader, so `tools/backtest.py` scores leave-one-launch-out on the comparables with the contest's
log-scale pinball loss; `tools/tune.py` does a coarse coordinate search of the parameters.

```sh
python tools/backtest.py path/to/p00.in          # all launches
python tools/backtest.py path/to/p00.in --big    # launches with 20k+ viewers in hour 1 (GTA VI's regime)
```

Leave-one-launch-out on the 37 comparables of the practice cases (lower is better):

| launches | this model | house persistence baseline |
| --- | --- | --- |
| all 37 | 0.391 | 0.930 |
| 20k+ viewers in hour 1 | 0.249 | 0.321 |

## License

MIT
