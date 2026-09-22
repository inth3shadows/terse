// Encode stdin JSON with compressmcp's "TerseJSON" codec and verify a lossless round-trip.
// Emits JSON to stdout, same contract as toon_encode.mjs:
//   {"payload": "<legend + compressed JSON>", "full": "<banner + payload>", "lossless": bool}
//
// Uses the package's own compress/decompress/formatOutput (pinned in package.json), so this
// measures compressmcp's real output rather than a reimplementation.
//
// TWO figures, because formatOutput() prepends a status banner to the thing it encodes:
//   payload — the `Keys:` legend plus the compressed JSON. The legend is part of the
//             encoding and is counted, exactly as terse's own dictionary legend is counted
//             against terse. This is the number comparable to §1's terse and TOON columns.
//   full    — payload plus the "[compressmcp: N->M tokens ...]" banner line. Analogous to
//             terse's primer, which terse also accounts for separately rather than folding
//             into its codec %.
import { compress, decompress } from "compressmcp/dist/compress/terse.js";
import { formatOutput } from "compressmcp/dist/compress/dictionary.js";
import { readFileSync } from "node:fs";

const raw = readFileSync(0, "utf8");

let obj;
try {
  obj = JSON.parse(raw);
} catch {
  // Not JSON at all. The caller must render this as `n/a`, never as a 0% tie —
  // compressmcp is a JSON codec, so "cannot encode" is a different fact from "tied".
  process.stdout.write(JSON.stringify({ json: false }));
  process.exit(0);
}

try {
  const result = compress(raw);
  const full = formatOutput(result);
  const nl = full.indexOf("\n");
  const payload = nl === -1 ? full : full.slice(nl + 1);
  const back = decompress(result.compressed, result.dictionary);
  const lossless = JSON.stringify(back) === JSON.stringify(obj);
  process.stdout.write(JSON.stringify({ json: true, payload, full, lossless }));
} catch (e) {
  process.stdout.write(JSON.stringify({ json: true, error: String(e).slice(0, 200) }));
}
