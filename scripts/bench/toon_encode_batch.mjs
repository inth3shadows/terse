// `toon_encode.mjs` for many payloads: one JSON object per stdin line,
//   {"id": ..., "json": "<raw JSON text>"}  ->  {"id": ..., "toon": "...", "lossless": bool}
// or {"id": ..., "error": "..."} when the encoder throws. Same pinned encoder and the same
// round-trip test.
import { encode, decode } from "@toon-format/toon";
import { createInterface } from "node:readline";

const rl = createInterface({ input: process.stdin });
for await (const line of rl) {
  if (!line.trim()) continue;
  const { id, json } = JSON.parse(line);
  try {
    const obj = JSON.parse(json);
    const toon = encode(obj);
    let lossless = false;
    try {
      lossless = JSON.stringify(decode(toon)) === JSON.stringify(obj);
    } catch { lossless = false; }
    process.stdout.write(JSON.stringify({ id, toon, lossless }) + "\n");
  } catch (err) {
    process.stdout.write(JSON.stringify({ id, error: String(err).slice(0, 300) }) + "\n");
  }
}
