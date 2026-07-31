# terse as a container: an MCP stdio proxy that compresses a downstream server's results.
#
# terse has NO tools of its own — it rewrites another server's `tools/call` results in
# place. So the image bundles a small stdlib-only downstream (examples/demo_mcp_server.py)
# as the default CMD, which is what makes the container answer `tools/list` at all instead
# of looking broken to a registry inspector. Calling `demo_orders` returns the compressed
# form of a 40-record order book (9019 -> 3187 chars on the local run, -65%), so the demo
# demonstrates the actual product rather than just proving the image starts.
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

RUN set -eux; \
    git config --global --add safe.directory /src; \
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
