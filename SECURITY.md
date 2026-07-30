# Security Policy

terse sits between an MCP client and its downstream servers, wrapping and rewriting
tool traffic. Its trust model: the client and the downstream servers it wraps are
trusted (they're config the user chose to run); a downstream server's *tool output*
is not — it can come from a hostile or compromised source (e.g. a URL fetched from a
repo-committed `.mcp.json`, per `transport.py`), so terse treats decoding and diffing
of tool results as adversarial input, not just plumbing. Config files terse writes or
touches (`.terse-peers*.json`, `.claude.json` stashes, backups) can carry API keys and
other credentials inherited from wrapped servers' `env` blocks; these are written
`0600` and excluded from lossy transforms, but that does not stop a `git add -A` in
project scope — see USAGE.md's guidance on `.gitignore`.

**Out of scope:** vulnerabilities in wrapped downstream MCP servers themselves (report
to their maintainers), and threats requiring local filesystem access an attacker
already has (terse does not attempt to defend against a compromised host).

## Reporting a Vulnerability

Open a [GitHub security advisory](https://github.com/inth3shadows/terse/security/advisories/new)
for this repository, or file a regular issue if the report doesn't need to stay
private. Please include a minimal repro (which command, which config shape) — terse
has no telemetry, so reports are the only signal maintainers get.
