// Encode stdin JSON to TOON and verify a lossless round-trip. Emits JSON to stdout:
//   {"toon": "<encoded text>", "lossless": true|false}
// Uses the official @toon-format/toon reference encoder/decoder (pinned in package.json),
// so the benchmark measures TOON's own output, not a reimplementation.
import { encode, decode } from "@toon-format/toon";
import { readFileSync } from "node:fs";

const raw = readFileSync(0, "utf8");
const obj = JSON.parse(raw);
const toon = encode(obj);
let lossless = false;
try {
  // Lossless == decode(encode(x)) is deep-equal to the original parsed value.
  lossless = JSON.stringify(decode(toon)) === JSON.stringify(obj);
} catch { lossless = false; }
process.stdout.write(JSON.stringify({ toon, lossless }));
