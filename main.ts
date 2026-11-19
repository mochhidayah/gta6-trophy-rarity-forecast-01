// main.ts — entry point for the GTA VI trophy rarity forecaster.
//
// Reads every <id>.in case from the input directory and writes the matching
// <id>.out forecast next to it in the output directory. Offline, standard
// library only. Run locally with:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, readdirSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join, basename, extname } from "node:path";
import { forecastCase } from "./forecast.ts";
import type { CaseData } from "./forecast.ts";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  if (!existsSync(INPUT_DIR)) {
    console.error(`input directory not found: ${INPUT_DIR}`);
    process.exit(1);
  }
  const files = readdirSync(INPUT_DIR)
    .filter((f) => extname(f) === ".in")
    .sort();
  if (files.length === 0) {
    console.error(`no .in cases in ${INPUT_DIR}`);
    process.exit(1);
  }
  for (const file of files) {
    const id = basename(file, ".in");
    try {
      const raw = readFileSync(join(INPUT_DIR, file), "utf8");
      const c = JSON.parse(raw) as CaseData;
      if (!c.trophies || !c.history || !c.target || !c.at) {
        console.error(`${id}: case is missing trophies, history, target or at; skipping`);
        continue;
      }
      const out = forecastCase(c);
      const text = JSON.stringify(out);
      if (text.length > 65536) {
        console.error(`${id}: output ${text.length} bytes exceeds the 65,536 cap; skipping`);
        continue;
      }
      writeFileSync(join(OUTPUT_DIR, `${id}.out`), text);
      console.log(`${id}: wrote ${out.trophies.length} trophies, ${text.length} bytes`);
    } catch (err) {
      console.error(`${id}: failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

main();
