// forecast.ts — structural trophy-rarity forecaster for GTA VI (PS5).
//
// Model, invented in this session:
//   Every trophy follows a saturating adoption curve in hours since launch,
//   p(t) = C * (1 - exp(-(t / tau) ^ 0.85)), where the ceiling C and the time
//   constant tau come from the trophy's own metadata (chapter / mission /
//   activity keywords, then tier and points). The curve is anchored to what
//   was actually observed: the forecast in logit space is shifted by the gap
//   between the last observed rate and the curve's value at the checkpoint.
//   A second forecast comes from the observed logit-velocity (exponentially
//   weighted regression over the reads), shrunk toward the velocity the
//   structural curve implies. The two forecasts are blended by horizon
//   (short horizon trusts velocity, long horizon trusts structure).
//   Quantile spread is built in logit space from structural uncertainty,
//   velocity uncertainty, horizon, and binomial sampling noise, so the band
//   widens the further ahead we look and the thinner the data is.

export interface TrophyMeta {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

export interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
}

export interface CaseData {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophyMeta[];
  history: HistoryRead[];
}

export interface Prior {
  ceiling: number; // long-run share of tracked players who ever earn it, 0..1
  tauHours: number; // adoption time constant, hours since launch
}

const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");
const LOGIT_MIN = -6.9; // ~0.1%
const LOGIT_MAX = 6.9; // ~99.9%

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export function logit(p: number): number {
  const c = clamp(p, 0.001, 0.999);
  return Math.log(c / (1 - c));
}

export function sigmoid(x: number): number {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

function hoursSinceLaunch(iso: string): number {
  return (Date.parse(iso) - RELEASE_MS) / 3_600_000;
}

// --- structural prior from trophy metadata (first matching rule wins) ---
export function priorFor(t: TrophyMeta): Prior {
  const text = `${t.name} ${t.description}`.toLowerCase();
  const has = (...words: string[]): boolean => words.some((w) => text.includes(w));
  if (has("chapter 1")) return { ceiling: 0.68, tauHours: 36 };
  if (has("chapter 2")) return { ceiling: 0.58, tauHours: 54 };
  if (has("chapter 3")) return { ceiling: 0.5, tauHours: 72 };
  if (has("chapter 4")) return { ceiling: 0.44, tauHours: 90 };
  if (has("final mission")) return { ceiling: 0.2, tauHours: 130 };
  if (has("100% completion")) return { ceiling: 0.025, tauHours: 500 };
  if (has("250 stunts")) return { ceiling: 0.05, tauHours: 350 };
  if (has("gold medal")) return { ceiling: 0.07, tauHours: 300 };
  if (has("drone race")) return { ceiling: 0.12, tauHours: 200 };
  if (has("stunt")) return { ceiling: 0.08, tauHours: 280 };
  if (has("win") && has("race")) return { ceiling: 0.14, tauHours: 200 };
  if (has("races")) return { ceiling: 0.12, tauHours: 220 };
  if (has("arcade")) return { ceiling: 0.22, tauHours: 150 };
  if (has("fish")) return { ceiling: 0.3, tauHours: 120 };
  if (has("activity")) return { ceiling: 0.18, tauHours: 160 };
  if (has("complete") && t.tier === "bronze") return { ceiling: 0.32, tauHours: 100 };
  if (t.tier === "platinum") return { ceiling: 0.01, tauHours: 600 };
  if (t.tier === "gold") return { ceiling: 0.08, tauHours: 260 };
  if (t.tier === "silver") return { ceiling: 0.12, tauHours: 200 };
  return { ceiling: 0.22, tauHours: 140 };
}

// Saturating adoption curve: share of players at t hours since launch.
export function structuralCurve(prior: Prior, tHours: number): number {
  const t = Math.max(0, tHours);
  const shape = 1 - Math.exp(-Math.pow(t / prior.tauHours, 0.85));
  return prior.ceiling * shape;
}

// Exponentially weighted linear regression of logit(rate) on time.
// Returns the slope (logit per hour), the effective sample size, and the
// fitted value at the checkpoint. Reads with no rate for this trophy are
// skipped; a zero rate is treated as half a tracked player.
export function estimateVelocity(
  times: number[],
  ratesPct: (number | null)[],
  players: number[],
  atHours: number,
): { slope: number; nEff: number; fittedAt: number } {
  const halfLife = 8; // hours; recent reads weigh far more than old ones
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < times.length; i++) {
    const r = ratesPct[i];
    if (r === null) continue;
    const p = r <= 0 ? 0.5 / Math.max(1, players[i]) / 100 : r / 100;
    const w = Math.exp(-(atHours - times[i]) / halfLife);
    const y = logit(p);
    sw += w;
    sx += w * times[i];
    sy += w * y;
    sxx += w * times[i] * times[i];
    sxy += w * times[i] * y;
  }
  if (sw <= 0) return { slope: 0, nEff: 0, fittedAt: 0 };
  const denom = sw * sxx - sx * sx;
  if (Math.abs(denom) < 1e-9) return { slope: 0, nEff: sw, fittedAt: sy / sw };
  const slope = (sw * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / sw;
  return { slope, nEff: sw, fittedAt: intercept + slope * atHours };
}

export interface TrophyForecast {
  id: number;
  q: number[]; // five percentages, non-decreasing
  medianPct: number;
  sigma: number;
}

export function forecastTrophy(
  trophy: TrophyMeta,
  series: (number | null)[],
  times: number[],
  players: number[],
  atHours: number,
  targetHours: number,
): TrophyForecast {
  const dt = Math.max(0, targetHours - atHours);
  const prior = priorFor(trophy);
  const lastPlayers = players.length ? players[players.length - 1] : 638;

  // Last observed rate (walk back past nulls).
  let lastRate: number | null = null;
  for (let i = series.length - 1; i >= 0; i--) {
    if (series[i] !== null) {
      lastRate = series[i] as number;
      break;
    }
  }
  if (lastRate === null) lastRate = 0;
  // A trophy everybody already has stays everybody's.
  if (lastRate >= 99.99) {
    return { id: trophy.id, q: [100, 100, 100, 100, 100], medianPct: 100, sigma: 0.05 };
  }

  const pLast = lastRate <= 0 ? 0.5 / Math.max(1, lastPlayers) / 100 : lastRate / 100;
  const logitLast = logit(pLast);

  // The ceiling is a prior, not a fact: if the trophy already sits above it,
  // the prior was wrong and the effective ceiling must sit above where we
  // stand, or the velocity forecast would be clamped below the last read.
  const effCeiling = Math.max(prior.ceiling, pLast * 1.5);
  // --- structural forecast, anchored to the observed level ---
  const curveAt = structuralCurve(prior, atHours);
  const curveTarget = structuralCurve(prior, targetHours);
  const anchorW = 0.85;
  const logitStruct =
    logit(curveTarget) + anchorW * (logitLast - logit(Math.max(curveAt, 0.0005)));

  // --- velocity forecast, shrunk toward the curve's implied velocity ---
  const vel = estimateVelocity(times, series, players, atHours);
  const priorVel =
    dt > 0
      ? (logit(Math.max(curveTarget, 0.0005)) - logit(Math.max(curveAt, 0.0005))) / dt
      : 0;
  // Prior counts as 1 effective read; real reads earn weight as they pile up.
  const k = 1;
  const vBlend = (vel.nEff * vel.slope + k * priorVel) / (vel.nEff + k);
  // Velocity decays as the rate closes on its ceiling (headroom shrinks),
  // and the velocity forecast may never pass 98% of the ceiling: a linear
  // logit extrapolation over days would otherwise overshoot saturation.
  const headroom = clamp(1 - pLast / effCeiling, 0.05, 1);
  const logitVel = clamp(
    logitLast + vBlend * dt * headroom * headroom,
    logitLast - 0.6,
    logit(0.98 * effCeiling),
  );

  // --- blend by horizon: short horizons trust velocity, long ones structure ---
  const wStruct = dt <= 12 ? 0.4 : dt >= 72 ? 0.65 : 0.4 + (0.25 * (dt - 12)) / 60;
  let logitMed = wStruct * logitStruct + (1 - wStruct) * logitVel;
  // Never forecast a collapse far below where we stand: the tracked base
  // grows, but a trophy's holders do not un-earn it; dilution is bounded.
  logitMed = Math.max(logitMed, logitLast - 0.6);
  logitMed = clamp(logitMed, LOGIT_MIN, LOGIT_MAX);

  // --- spread in logit space, growing with the horizon ---
  // At a 30-minute horizon almost nothing can move, so the band is just
  // sampling noise; at a 5-day horizon the structural and velocity
  // uncertainty take over. The horizon factor ramps with a ~24h constant.
  const earnedEst = Math.max(1, pLast * lastPlayers);
  const sigmaBinom = clamp(1 / Math.sqrt(earnedEst), 0.05, 0.8);
  const sigmaVel = Math.abs(vBlend) * dt * 0.25;
  const sigmaSparse = 0.25 * (1 - vel.nEff / (vel.nEff + 3));
  // The binomial term is uncertainty about the latent adoption probability:
  // it matters for the counts five days out, not for the next half hour,
  // when the truth is just the count we already see. So it lives in the
  // long component, not the short one.
  const sigmaShort = 0.05;
  const sigmaLong = Math.sqrt(
    0.3 * 0.3 + sigmaBinom * sigmaBinom + sigmaVel * sigmaVel + sigmaSparse * sigmaSparse,
  );
  const horizonFrac = dt / (dt + 24);
  const sigma = clamp(
    Math.sqrt(
      sigmaShort * sigmaShort + (sigmaLong * sigmaLong - sigmaShort * sigmaShort) * horizonFrac,
    ),
    0.08,
    1.2,
  );

  const z = [-1.64485, -0.67449, 0, 0.67449, 1.64485];
  const q = z.map((zi) => {
    const pct = sigmoid(logitMed + zi * sigma) * 100;
    return Math.round(clamp(pct, 0.01, 99.99) * 100) / 100;
  });
  // Enforce non-decreasing after rounding.
  for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
  const medianPct = Math.round(sigmoid(logitMed) * 10000) / 100;
  return { id: trophy.id, q, medianPct, sigma };
}

export function forecastCase(c: CaseData): { trophies: { id: number; q: number[] }[] } {
  const atHours = hoursSinceLaunch(c.at);
  const targetHours = hoursSinceLaunch(c.target);
  const times = c.history.map((r) => hoursSinceLaunch(r.at));
  const players = c.history.map((r) => r.players_tracked);
  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const series = c.history.map((r) => r.rates[i]);
    const f = forecastTrophy(c.trophies[i], series, times, players, atHours, targetHours);
    rows.push({ id: f.id, q: f.q });
  }
  return { trophies: rows };
}
