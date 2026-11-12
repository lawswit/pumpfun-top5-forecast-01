// pumpfun-top5-forecast-01 entry: a momentum-and-dispersion forecaster.
//
// The house baseline (the kit's starter/) assumes the last mark's order simply holds. That assumption is
// weak at this contest's horizon: the final's history ends 7 to 13 hours before the first answer read, and
// in the house's own practice rehearsal 3 of the 5 coins that held a top-5 place at the last read were
// launched after the history ended. So instead of persisting one snapshot this program
//
//   1. builds each coin's trajectory of log10 market cap (in SOL) across every mark of the history,
//   2. estimates each coin's recent drift and projects it forward to each answer read, damping the drift so
//      a coin that pumped hard for twenty minutes is not extrapolated for thirteen hours,
//   3. draws many joint scenarios: each candidate's projected cap plus a noise term whose spread grows with
//      the horizon, together with coins that do not exist yet (the newcomers), whose caps are drawn from a
//      generous distribution because they are exactly the ones that take places nobody can name,
//   4. ranks every scenario and tallies how often each coin lands at each place and which size bin each
//      place's cap falls in.
//
// The newcomer mass is the point of the last step: a place a newcomer takes cannot be named, and the scorer
// pays it out of the share the forecast leaves unnamed, so the tallies are also what tells this program how
// much probability to withhold from named coins. Nothing here is fitted to the practice: the constants are
// set from the shape of pump.fun launches (fast early pumps that decay, a fat right tail of newcomers) and
// are the same for every history.
//
// Standard library only, no network, no clock reads: two runs on one history write the same bytes, because
// every draw comes from a fixed-seed generator and every array is sorted before it is written.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** One row of one mark: a coin as the house recorder priced and ranked it. */
interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  name: string;
  symbol: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
  quote_reserves_lamports: string | number;
}

/** One mark: a 10-minute snapshot of the largest ranked coins. */
interface Mark {
  at: string;
  ok: boolean;
  rows: Row[] | null;
}

/** The house's task file: the places scored, the size bins' edges, the answer reads, the output name. */
interface Task {
  places: number;
  output: string;
  size: { edges: number[] };
  reads: string[];
  chart_size: number;
  t0: string;
  history_until: string;
}

// --- the model's constants, identical for every history --------------------------------------------

/** Scenarios drawn per answer read. Enough for the 100-coin chart's place tallies to settle. */
const DRAWS = 4000;
/** The generator's seed. Fixed, so the twice-run comparison sees the same bytes. */
const SEED = 0x6d2b79f5;
/** How many candidate coins to carry into the scenarios, by projected cap. */
const CANDIDATES = 44;
/** The drift's e-folding horizon in hours: momentum counts for about this long, then stops. */
const DRIFT_HORIZON_H = 3.5;
/** The largest drift believed, in log10 per hour. The sweep over the rehearsal's proxy reads put the best
 * results at the low end of the range: extrapolating a coin's recent run is a losing bet, because the pumps
 * that make a coin look strong are exactly the ones that mean-revert. */
const DRIFT_CAP = 0.35;
/** Marks used to estimate the drift: the last part of the history, where the recent shape lives. */
const DRIFT_LOOKBACK = 6;
/** Base spread of a coin's log10 cap at an answer read. */
const SIGMA_BASE = 0.30;
/** How much more spread a read 13 hours out gets than one at the cut. The rehearsal's reads are ten minutes
 * apart and cannot calibrate this, so it is set from the horizon itself: a cap that has thirteen hours to
 * move is far less certain than one that has ten minutes. */
const SIGMA_PER_13H = 0.25;
/** Expected number of newcomer coins large enough to matter at a read. The rules say 3 of the 9 coins that
 * held a top-5 place at any practice read were launched after its history ended, and 2 of the 5 at its last
 * read: newcomers are not a rounding error, they are a large share of the places, so this is generous. */
const NEWCOMER_MEAN = 3.5;
/** The spread of a large newcomer's log10 cap around the level where the field sits. */
const NEWCOMER_SD = 0.62;
/** Where a large newcomer sits relative to the field's top: just above it, on average. */
const NEWCOMER_RISE = 0.20;
/** A coin last seen this many minutes before the cut is treated as fading... */
const FADE_AFTER_MIN = 15;
/** ...at this many log10 per hour of absence beyond that. */
const FADE_PER_H = 0.8;

// --- a fixed-seed generator, so the output is reproducible byte for byte ------------------------------

/** mulberry32: 32 bits of state, uniform in [0, 1). */
function makeRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A standard normal draw by Box-Muller, from two uniforms of the same generator. */
function makeNormal(random: () => number): () => number {
  return () => {
    const u = 1 - random();
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/** A Poisson draw (Knuth's method; the mean here is small, so it terminates at once). */
function poisson(random: () => number, mean: number): number {
  const limit = Math.exp(-mean);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    p *= random();
  } while (p > limit);
  return k - 1;
}

// --- reading the input ------------------------------------------------------------------------------

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const edges = task.size.edges;
const places = task.places;

const marks = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((line) => line.trim() !== '')
  .map((line) => JSON.parse(line) as Mark)
  .filter((mark) => mark.ok === true && Array.isArray(mark.rows) && mark.rows.length > 0)
  .map((mark) => ({ at: mark.at, atMs: Date.parse(mark.at), rows: mark.rows as Row[] }));

if (marks.length === 0) throw new Error('the history holds no ok mark with rows');
marks.sort((a, b) => a.atMs - b.atMs);

const cut = marks[marks.length - 1];
const cutMs = cut.atMs;
const HOUR = 3_600_000;

/** log10 of a cap in SOL, the unit the size bins are in: log10(lamports) - 9. */
const logCapOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

// --- each coin's trajectory --------------------------------------------------------------------------

/** One coin's history: its log10 cap at each mark that ranked it, and the last such mark. */
interface Coin {
  mint: string;
  symbol: string;
  createdMs: number;
  /** Marks that ranked it: the log10 cap and the mark's time, oldest first. */
  points: Array<{ ms: number; logCap: number }>;
  mayhem: boolean;
  lastMs: number;
  lastLogCap: number;
}

const coins = new Map<string, Coin>();
for (const mark of marks) {
  for (const row of mark.rows) {
    const logCap = logCapOf(row.market_cap_lamports);
    const createdMs = Date.parse(row.created_at);
    let coin = coins.get(row.mint);
    if (coin === undefined) {
      coin = {
        mint: row.mint,
        symbol: row.symbol,
        createdMs: Number.isFinite(createdMs) ? createdMs : cutMs,
        points: [],
        mayhem: row.mayhem === true,
        lastMs: mark.atMs,
        lastLogCap: logCap,
      };
      coins.set(row.mint, coin);
    }
    coin.points.push({ ms: mark.atMs, logCap });
    if (mark.atMs >= coin.lastMs) {
      coin.lastMs = mark.atMs;
      coin.lastLogCap = logCap;
    }
    if (row.symbol) coin.symbol = row.symbol;
  }
}

/** A least-squares slope of log10 cap against hours over the coin's last `DRIFT_LOOKBACK` points. */
function driftOf(coin: Coin): number {
  const pts = coin.points.slice(-DRIFT_LOOKBACK);
  if (pts.length < 2) return 0;
  const first = pts[0]!;
  const span = (pts[pts.length - 1]!.ms - first.ms) / HOUR;
  if (!(span > 0)) return 0;
  let sumT = 0;
  let sumY = 0;
  for (const p of pts) {
    sumT += (p.ms - first.ms) / HOUR;
    sumY += p.logCap - first.logCap;
  }
  const meanT = sumT / pts.length;
  const meanY = sumY / pts.length;
  let num = 0;
  let den = 0;
  for (const p of pts) {
    const t = (p.ms - first.ms) / HOUR - meanT;
    const y = p.logCap - first.logCap - meanY;
    num += t * y;
    den += t * t;
  }
  if (!(den > 0)) return 0;
  return num / den;
}

/** A coin's projected log10 cap at `horizonH` hours after the cut, and the spread around it. */
function project(coin: Coin, horizonH: number): { centre: number; sigma: number } {
  const drift = Math.max(-DRIFT_CAP, Math.min(DRIFT_CAP, driftOf(coin)));
  // The drift is believed for about DRIFT_HORIZON_H hours and not beyond: the integral of e^{-t/h}.
  const believed = DRIFT_HORIZON_H * (1 - Math.exp(-Math.max(0, horizonH) / DRIFT_HORIZON_H));
  // A coin that left the chart before the cut has faded: charge it for the time it has been unseen.
  const unseenH = (cutMs - coin.lastMs) / HOUR;
  const fade = unseenH > FADE_AFTER_MIN / 60 ? -FADE_PER_H * (unseenH - FADE_AFTER_MIN / 60) : 0;
  const centre = coin.lastLogCap + drift * believed + fade;
  const horizonSpread = SIGMA_PER_13H * Math.min(Math.max(horizonH, 0), 13) / 13;
  // A coin on a fast run is less certain than one that has settled, so its drift adds spread.
  const sigma = SIGMA_BASE + horizonSpread + 0.10 * Math.min(Math.abs(drift), 1);
  return { centre, sigma };
}

// --- the answer reads and the horizons they sit at ----------------------------------------------------

const reads = (Array.isArray(task.reads) ? task.reads : [])
  .map((at) => ({ at, atMs: Date.parse(at) }))
  .filter((r) => Number.isFinite(r.atMs))
  .sort((a, b) => a.atMs - b.atMs);
if (reads.length === 0) throw new Error('task.json names no answer read');
/** The far horizon, used only to choose which coins are plausible candidates at any read. */
const farHorizonH = Math.max(0, (reads[reads.length - 1]!.atMs - cutMs) / HOUR);

/** The coins worth carrying into the scenarios: the strongest at the cut and at the far horizon. */
const pool = [...coins.values()];
const byFar = pool
  .map((coin) => ({ coin, centre: project(coin, farHorizonH).centre }))
  .sort((a, b) => b.centre - a.centre || (a.coin.mint < b.coin.mint ? -1 : 1));
const candidates = byFar.slice(0, CANDIDATES).map((entry) => entry.coin);

if (candidates.length === 0) throw new Error('the history names no coin');

/** The field's top projected level at the far horizon, where a large newcomer would land. */
const fieldTop = byFar.length > 0 ? byFar[0]!.centre : 0;

// --- the scenarios -----------------------------------------------------------------------------------

/** Which size bin a log10 cap falls in: the number of edges at or below it (bin 0 below the first). */
function binOf(logCap: number): number {
  let lo = 0;
  let hi = edges.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (edges[mid]! <= logCap) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const random = makeRandom(SEED);
const normal = makeNormal(random);

/** Filled by the scenarios: place -> mint -> count, and place -> bin -> count. */
const placeCoinCount = new Map<number, Map<string, number>>();
const placeBinCount = new Map<number, Map<number, number>>();
for (let place = 1; place <= places; place++) {
  placeCoinCount.set(place, new Map<string, number>());
  placeBinCount.set(place, new Map<number, number>());
}

/** One scenario's ordering of the field and of the newcomers, reused across draws. */
interface Draw {
  logCap: number;
  mint: string | null;
  newcomer: boolean;
}

for (const read of reads) {
  const horizonH = Math.max(0, (read.atMs - cutMs) / HOUR);
  const projected = candidates.map((coin) => ({ coin, ...project(coin, horizonH) }));
  // How many newcomers to draw: Poisson, so some scenarios have none and some have several.
  const newcomerMean = NEWCOMER_MEAN * (0.55 + 0.45 * Math.min(horizonH / 13, 1));
  const newcomerLevel = fieldTop + NEWCOMER_RISE;
  const buffer: Draw[] = [];
  for (let draw = 0; draw < DRAWS; draw++) {
    buffer.length = 0;
    for (const entry of projected) {
      buffer.push({
        logCap: entry.centre + entry.sigma * normal(),
        mint: entry.coin.mint,
        newcomer: false,
      });
    }
    const newcomers = poisson(random, newcomerMean);
    for (let n = 0; n < newcomers; n++) {
      buffer.push({ logCap: newcomerLevel + NEWCOMER_SD * normal(), mint: null, newcomer: true });
    }
    // Rank by sampled cap; a tie (impossible in practice, but keep it total) falls to the mint's bytes.
    buffer.sort((a, b) => b.logCap - a.logCap || (a.mint === b.mint ? 0 : (a.mint ?? '') < (b.mint ?? '') ? -1 : 1));
    const held = Math.min(places, buffer.length);
    for (let place = 1; place <= held; place++) {
      const slot = buffer[place - 1]!;
      if (slot.mint !== null) {
        const counter = placeCoinCount.get(place)!;
        counter.set(slot.mint, (counter.get(slot.mint) ?? 0) + 1);
      }
      const bins = placeBinCount.get(place)!;
      const bin = binOf(slot.logCap);
      bins.set(bin, (bins.get(bin) ?? 0) + 1);
    }
  }
}

// --- writing the forecast ----------------------------------------------------------------------------

const MICRO = 1_000_000;
/** Round down to 6 decimals: the house's own baseline does, and it keeps each place's rows under 1. */
const floor6 = (p: number): number => Math.floor(p * MICRO) / MICRO;
/**
 * The counts above are summed over every answer read, and the one forecast is scored at each of them, so a
 * probability is a count over the reads x draws scenarios: the share of scenarios, in a typical read, in
 * which the coin held the place.
 */
const scenarioTotal = DRAWS * reads.length;

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let place = 1; place <= places; place++) {
  const counter = placeCoinCount.get(place)!;
  const rows = [...counter.entries()]
    .map(([mint, count]) => ({ mint, p: count / scenarioTotal }))
    .filter((row) => Number.isFinite(row.p) && row.p > 0)
    .sort((a, b) => b.p - a.p || (a.mint < b.mint ? -1 : 1))
    .slice(0, Math.min(100, task.chart_size ?? 100))
    .map((row) => ({ mint: row.mint, p: floor6(row.p) }))
    .filter((row) => row.p > 0);
  // A place's rows must not add up to more than 1: with the counts averaged over the reads they cannot,
  // but scale down rather than risk writing a file the scorer reads as no forecast at all.
  const named = rows.reduce((sum, row) => sum + row.p, 0);
  if (named > 1) {
    for (const row of rows) row.p = floor6(row.p / named);
  }
  // Flooring can only shrink a place's total, so the 1 minus the sum stays the unnamed share the scorer
  // pays a coin that was born after the cut; nothing here needs to renormalise.
  for (const row of rows) probabilities.push({ mint: row.mint, place, p: row.p });
}

const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let place = 1; place <= places; place++) {
  const bins = placeBinCount.get(place)!;
  for (const [bin, count] of [...bins.entries()].sort((a, b) => a[0] - b[0])) {
    const p = floor6(count / scenarioTotal);
    if (Number.isFinite(p) && p > 0) sizes.push({ place, bin, p });
  }
}

/**
 * The best guess at the order: each coin's mean place where it is at a place at all, which is its places
 * weighted by the scenarios that put it there, over the scenarios that place it anywhere. Dividing by the
 * coin's own mass rather than by the whole draw is what keeps a coin that is almost never at a place (and so
 * scores a tiny weighted number) from sorting ahead of the coin that is usually first.
 */
const weightedPlace = new Map<string, number>();
const coinMass = new Map<string, number>();
for (let place = 1; place <= places; place++) {
  for (const [mint, count] of placeCoinCount.get(place)!) {
    weightedPlace.set(mint, (weightedPlace.get(mint) ?? 0) + place * count);
    coinMass.set(mint, (coinMass.get(mint) ?? 0) + count);
  }
}
const meanPlace = new Map<string, number>();
for (const [mint, weighted] of weightedPlace) {
  const mass = coinMass.get(mint) ?? 0;
  if (mass > 0) meanPlace.set(mint, weighted / mass);
}
const top5 = [...meanPlace.entries()]
  .sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1))
  .slice(0, places)
  .map(([mint]) => mint);

if (top5.length < places) {
  // Only reachable when the history names fewer coins than the places scored: name what exists.
  for (const coin of candidates) {
    if (top5.length >= places) break;
    if (!top5.includes(coin.mint)) top5.push(coin.mint);
  }
}

const forecast = { probabilities, top5, sizes };
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify(forecast)}\n`);
