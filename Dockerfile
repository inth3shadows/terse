# terse as a container: an MCP stdio proxy that compresses a downstream server's results.
#
# terse has NO tools of its own — it rewrites another server's `tools/call` results in
# place. So the image bundles a small stdlib-only downstream (examples/demo_mcp_server.py)
# as the default CMD, which is what makes the container answer `tools/list` at all instead
# of looking broken when it is run bare. Calling `demo_orders` returns the compressed form
# of a 40-record order book (9019 -> 3187 chars on the local run, -65%), so the demo
# demonstrates the actual product rather than just proving the image starts.
#
# This image is NOT what the Glama registry builds, despite #194's commit message saying it
# was added "so the Glama listing builds". Glama generates its own spec (debian:trixie-slim,
# `uv sync`, wrapped in mcp-proxy) and ignores this file; `glama.json`'s schema admits
# exactly one key, `maintainers`, so there is no way to point the registry at a Dockerfile.
# What this file IS for is the documented `docker run` path below -- and the `docker` job in
# .github/workflows/tests.yml is the only thing keeping it from rotting unnoticed, which it
# did for three weeks after #194. Verified 2026-08-20.
#
# Point it at a real server to use the image for work — CMD is the downstream command:
#   docker run -i --rm terse uvx some-mcp-server
#   docker run -i --rm -v "$PWD/policy.json:/policy.json:ro" terse \
#     --policy /policy.json -- uvx some-mcp-server        # (see the ENTRYPOINT note below)
#
# `-i` is not optional: MCP stdio means the protocol IS stdin/stdout.

# ---- build stage: needs the git history, the runtime image must not carry it ----------
FROM python:3.12-slim AS builder

# The version is the GIT TAG (hatch-vcs), so a build context WITHOUT `.git` fails hard
# rather than silently shipping 0.0.0 — see [tool.hatch.version] in pyproject.toml. A
# normal `git clone` build context has it. Building from a source tarball does not, so
# pass the version explicitly: `--build-arg TERSE_VERSION=0.3.1`.
# Note the env var is the UNSCOPED one. The dist-name-scoped
# `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TERSE_MCP` spelling does NOT work here: hatch-vcs
# does not hand the distribution name down to setuptools-scm, so the scoped variable
# never matches and the build fails with the same LookupError. Verified 2026-07-30.
ARG TERSE_VERSION=

# setuptools-scm shells out to `git describe`, and python:*-slim has no git binary — so
# a `.git` in the build context is necessary but NOT sufficient. Without this the build
# fails with the same "unable to detect version" LookupError as a missing .git, which
# points at the wrong cause entirely. Builder stage only; the runtime image has no git.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends git; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Deliberately no `git config --global --add safe.directory`: git resolves the repository
# even for a `--global` write, so with a worktree-pointer `.git` (the layout this repo is
# developed in) it exits 128 on a path that does not exist in the container — failing the
# build BEFORE the TERSE_VERSION branch could rescue it. /src is root-owned and built as
# root, so there is no ownership check to appease in the first place.
RUN set -eux; \
    if [ -n "${TERSE_VERSION}" ]; then export SETUPTOOLS_SCM_PRETEND_VERSION="${TERSE_VERSION}"; fi; \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---- runtime stage --------------------------------------------------------------------
FROM python:3.12-slim

# tiktoken lazily DOWNLOADS its BPE table the first time an encoding is loaded. Registry
# sandboxes run each server in a throwaway microVM with its own ephemeral network, so a
# first-call fetch is a hang waiting to happen. Pin the cache to a fixed path and warm it
# at build time, below, so the running container never needs the network.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

# `uvx` so the documented `docker run -i --rm terse uvx some-mcp-server` actually works:
# the runtime stage is otherwise python-slim plus one wheel, with no launcher on PATH at
# all, and most real MCP servers are uvx-launched. Pinned, not `:latest` — the tag is the
# only thing keeping this reproducible. Node/`npx` is deliberately NOT installed (~150MB
# for a second ecosystem); for an npx-launched server, FROM this image and add Node.
# Note a uvx downstream needs the network at run time to resolve the package; only the
# bundled demo path is network-free.
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /usr/local/bin/

COPY --from=builder /wheels /wheels
RUN set -eux; \
    pip install --no-cache-dir /wheels/*.whl; \
    rm -rf /wheels; \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"; \
    chmod -R a+rX "${TIKTOKEN_CACHE_DIR}"; \
    useradd --create-home --uid 10001 terse

COPY examples/demo_mcp_server.py /app/examples/demo_mcp_server.py

USER terse
WORKDIR /app

# Docker does NOT set HOME from /etc/passwd on a `USER` switch — it stays /root, which the
# terse user cannot write. The savings ledger defaults to $XDG_STATE_HOME, falling back to
# ~/.local/state (stats.py), so without this the ledger silently writes nowhere.
ENV HOME=/home/terse

# Everything after `--` is the downstream server command, which is exactly what CMD
# supplies. To pass terse's own flags you must restate the entrypoint:
#   docker run -i --rm --entrypoint terse <image> proxy --debug -- uvx some-mcp-server
ENTRYPOINT ["terse", "proxy", "--"]
CMD ["python", "/app/examples/demo_mcp_server.py"]
