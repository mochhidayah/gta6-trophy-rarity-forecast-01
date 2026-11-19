// backtest.ts — local validation for the trophy forecaster.
//
// Two checks:
//   A) Rolling-origin on the real practice cases (p00/p01): hold out the last
//      reads as pseudo-targets, score mine vs persistence vs a trend
//      baseline. Launch-night reads are flat, so expect near-ties here.
//   B) Synthetic later-cut test: a progression simulator (different curve
//      family and parameters from the forecaster's priors, plus player-base
//      growth and binomial noise) generates 2-day histories; cases are cut
//      at day 1 / 1.5 / 2 with the truth at day 7. This is where a structural
//      forecaster must beat persistence and trend.
//
// Pinball loss is on the logit scale: for true rate y (percent) and forecast
// quantile q at level tau, loss = (logit(y) - logit(q)) * tau when y >= q,
// else (logit(q) - logit(y)) * (1 - tau); averaged over trophies and levels.

import { readFileSync } from "node:fs";
import { forecastCase, forecastTrophy, logit, sigmoid, clamp, priorFor } from "./forecast.ts";
import type { CaseData, TrophyMeta, HistoryRead } from "./forecast.ts";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.64485, -0.67449, 0, 0.67449, 1.64485];

function pinballOne(truthPct: number, qPct: number, tau: number): number {
  // Rates are percents (0..100); the logit scale needs fractions.
  const y = logit(clamp(truthPct, 0.01, 99.99) / 100);
  const q = logit(clamp(qPct, 0.01, 99.99) / 100);
  return y >= q ? (y - q) * tau : (q - y) * (1 - tau);
}

export function pinball(truthPct: number[], forecastQs: number[][]): number {
  let total = 0;
  let n = 0;
  for (let i = 0; i < truthPct.length; i++) {
    for (let l = 0; l < LEVELS.length; l++) {
      total += pinballOne(truthPct[i], forecastQs[i][l], LEVELS[l]);
      n++;
    }
  }
  return total / n;
}

// --- baselines, implemented here for comparison only ---
export function persistenceForecast(c: CaseData): number[][] {
  const dDays = c.horizon_minutes / 1440;
  const spread = 0.5 * Math.sqrt(dDays);
  const last = c.history[c.history.length - 1];
  return c.trophies.map((_, i) => {
    const r = last.rates[i] ?? 0;
    const base = logit(clamp(r, 0.01, 99.99) / 100);
    return Z.map((z) => {
      const pct = sigmoid(base + z * spread) * 100;
      return Math.round(clamp(pct, 0.01, 99.99) * 100) / 100;
    });
  });
}

export function trendForecast(c: CaseData): number[][] {
  // Linear logit extrapolation of the last two reads' slope, same spread.
  const dDays = c.horizon_minutes / 1440;
  const spread = 0.5 * Math.sqrt(dDays);
  const h = c.history;
  const t1 = Date.parse(h[h.length - 1].at);
  const t0 = Date.parse(h[h.length - 2].at);
  const dtH = Math.max(0.5, (t1 - t0) / 3_600_000);
  const dtTargetH = c.horizon_minutes / 60;
  return c.trophies.map((_, i) => {
    const r1 = h[h.length - 1].rates[i] ?? 0;
    const r0 = h[h.length - 2].rates[i] ?? r1;
    const slope = (logit(clamp(r1, 0.01, 99.99) / 100) - logit(clamp(r0, 0.01, 99.99) / 100)) / dtH;
    const med = clamp(logit(clamp(r1, 0.01, 99.99) / 100) + slope * dtTargetH, -6.9, 6.9);
    return Z.map((z) => {
      const pct = sigmoid(med + z * spread) * 100;
      return Math.round(clamp(pct, 0.01, 99.99) * 100) / 100;
    });
  });
}

// --- A) rolling-origin on the real practice cases ---
function rollingOrigin(): void {
  console.log("=== A) rolling-origin on real practice cases ===");
  for (const name of ["p00", "p01"]) {
    const full = JSON.parse(readFileSync(`input/${name}.in`, "utf8")) as CaseData;
    const h = full.history;
    for (let cut = 3; cut < h.length; cut++) {
      const sub: CaseData = {
        ...full,
        at: h[cut - 1].at,
        target: h[cut].at,
        horizon_minutes: (Date.parse(h[cut].at) - Date.parse(h[cut - 1].at)) / 60000,
        history: h.slice(0, cut),
      };
      const truth = h[cut].rates.map((v) => v ?? 0);
      const mine = forecastCase(sub).trophies.map((r) => r.q);
      const pers = persistenceForecast(sub);
      const trend = trendForecast(sub);
      console.log(
        `${name} cut@${cut} (target ${h[cut].at.slice(11, 16)}): ` +
          `mine=${pinball(truth, mine).toFixed(4)} ` +
          `persistence=${pinball(truth, pers).toFixed(4)} ` +
          `trend=${pinball(truth, trend).toFixed(4)}`,
      );
    }
  }
}

// --- B) synthetic later-cut test ---
// A different progression model from the forecaster's priors: per-trophy
// logistic adoption in log-time, ceilings and speeds drawn at random within
// tier bands, a growing tracked base, and binomial sampling noise. The
// forecaster never sees these parameters.
interface SimTrophy {
  meta: TrophyMeta;
  ceiling: number;
  midHours: number; // time at which the trophy reaches half its ceiling
  steep: number;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildSimTrophies(metas: TrophyMeta[], rng: () => number): SimTrophy[] {
  return metas.map((m) => {
    const band =
      m.tier === "platinum"
        ? { c: [0.002, 0.02], mid: [300, 900], s: [0.5, 1.0] }
        : m.tier === "gold"
          ? { c: [0.02, 0.25], mid: [80, 250], s: [0.6, 1.2] }
          : m.tier === "silver"
            ? { c: [0.03, 0.35], mid: [50, 200], s: [0.6, 1.2] }
            : { c: [0.08, 0.8], mid: [20, 160], s: [0.6, 1.3] };
    // Story-order signal: earlier trophy ids in the list progress faster.
    const orderBoost = 1 - (m.id / metas.length) * 0.4;
    return {
      meta: m,
      ceiling: band.c[0] + rng() * (band.c[1] - band.c[0]),
      midHours: (band.mid[0] + rng() * (band.mid[1] - band.mid[0])) * orderBoost,
      steep: band.s[0] + rng() * (band.s[1] - band.s[0]),
    };
  });
}

function simRate(st: SimTrophy, tHours: number): number {
  // Logistic in log-time: p = C / (1 + (t/mid)^-steep).
  const x = Math.pow(Math.max(tHours, 0.5) / st.midHours, st.steep);
  return st.ceiling * (x / (1 + x));
}

function syntheticTest(): void {
  console.log("=== B) synthetic later-cut test (2-day history, truth at day 7) ===");
  const base = JSON.parse(readFileSync("input/p01.in", "utf8")) as CaseData;
  const RELEASE = Date.parse("2026-11-19T00:00:00Z");
  const truthHours = 168; // 2026-11-26T00:00:00Z
  // Three simulator draws so a tuning choice is not fitted to one world.
  const seeds = [20261119, 7, 42];
  for (const seed of seeds) {
    const rng = mulberry32(seed);
    const sims = buildSimTrophies(base.trophies, rng);
    const truth = sims.map((s) => simRate(s, truthHours) * 100);
    for (const cutHours of [24, 36, 48]) {
    // Reads every 30 min from t=0 to the cut.
    const reads: HistoryRead[] = [];
    let players = 638;
    for (let t = 0; t <= cutHours; t += 0.5) {
      // Tracked base grows fast early, then saturates toward ~60k.
      players = Math.round(638 + (60000 - 638) * (1 - Math.exp(-t / 30)));
      const rates = sims.map((s) => {
        const p = simRate(s, Math.max(t, 0.5));
        const earned = Math.round(p * players);
        // Binomial noise, then Exophase's 2-decimal rounding.
        const noisy = Math.round((earned + (rng() - 0.5) * Math.sqrt(Math.max(earned, 1))) / players * 10000) / 100;
        return clamp(noisy, 0, 100);
      });
      const at = new Date(RELEASE + t * 3_600_000).toISOString().replace(".000Z", "Z");
      reads.push({
        slot: at,
        at,
        players_tracked: players,
        trophies_earned: rates.reduce((a, r) => a + Math.round((r / 100) * players), 0),
        rates,
      });
    }
    const atIso = new Date(RELEASE + cutHours * 3_600_000).toISOString().replace(".000Z", "Z");
    const c: CaseData = {
      ...base,
      at: atIso,
      target: "2026-11-26T00:00:00Z",
      horizon_minutes: (truthHours - cutHours) * 60,
      history: reads,
    };
    const mine = forecastCase(c).trophies.map((r) => r.q);
    const pers = persistenceForecast(c);
    const trend = trendForecast(c);
    const lm = pinball(truth, mine);
    const lp = pinball(truth, pers);
    const lt = pinball(truth, trend);
    console.log(
      `cut@${cutHours}h (horizon ${(truthHours - cutHours).toFixed(0)}h): ` +
        `mine=${lm.toFixed(4)} persistence=${lp.toFixed(4)} trend=${lt.toFixed(4)} ` +
        `(mine beats persistence by ${((lp - lm) / lp * 100).toFixed(1)}%, trend by ${((lt - lm) / lt * 100).toFixed(1)}%)`,
    );
    // Sanity: median forecast for a few trophies at the 48h cut, first seed.
    if (cutHours === 48 && seed === 20261119) {
      for (const i of [0, 2, 6, 13, 14]) {
        const f = forecastCase(c).trophies[i];
        console.log(
          `  trophy ${base.trophies[i].id} ${base.trophies[i].name.slice(0, 28)}: ` +
            `truth=${truth[i].toFixed(2)}% median=${f.q[2].toFixed(2)}% ` +
            `q05=${f.q[0].toFixed(2)} q95=${f.q[4].toFixed(2)}`,
        );
      }
    }
    }
  }
}

rollingOrigin();
syntheticTest();
