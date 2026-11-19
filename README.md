[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mochhidayah/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mochhidayah/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast how many GTA VI players earn each trophy in week one.

Forecasts, for each of the 55 GTA VI PS5 trophies, five quantiles (5%, 25%,
50%, 75%, 95%) of the share of tracked players who will have earned it at
the case's target (2026-11-26 00:00 UTC), from the Exophase reads in the
case's history. Runs offline, Node 22, standard library only.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory and writes `<id>.out` next to it
in the output directory: `{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}`.

## Method

`forecast.ts` — a structural forecaster, written from scratch in this
session:

- Every trophy follows a saturating adoption curve in hours since launch,
  `p(t) = C * (1 - exp(-(t/tau)^0.85))`, with the ceiling `C` and time
  constant `tau` from the trophy's own metadata (chapter / mission / activity
  keywords first, then tier and points).
- The curve is anchored to the observed level: the logit forecast is shifted
  by the gap between the last observed rate and the curve's value at the
  checkpoint.
- A second forecast comes from the observed logit-velocity (exponentially
  weighted regression over the reads, 8h half-life), shrunk toward the
  velocity the structural curve implies, and damped as the rate closes on
  its ceiling so a linear extrapolation cannot overshoot saturation.
- The two forecasts blend by horizon (short horizons trust velocity, long
  ones trust structure). Quantile spread is built in logit space from
  structural uncertainty, velocity uncertainty, horizon, and binomial
  sampling noise, so the band is tight at a 30-minute horizon and wide at
  five days.

`backtest.ts` — local validation: rolling-origin on the practice cases, plus
a synthetic later-cut test (a different progression simulator, 2-day
histories, truth at day 7, three seeds) scoring pinball loss on the logit
scale against persistence and a trend baseline. See SUBMISSION.md for the
numbers.

## Files

- `main.ts` — entry point (I/O driver).
- `forecast.ts` — the model.
- `backtest.ts` — validation harness (not used at run time).
- `main.py` — the starter's Python copy of the persistence baseline,
  untouched except a build note.
