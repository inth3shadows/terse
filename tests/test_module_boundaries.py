"""The one architectural boundary terse enforces in code rather than in prose (#237).

`multiproxy.py` is a router — scatter/gather across peers, broadcast timeouts, id
rewriting, collision-aware naming, listing merges. None of it is compression. terse's
product is a lossless-first ENCODER; the router is a second thing living in the same repo,
and #237's rule is that the two do not blend:

    New optimization logic does not land in `multiproxy.py`. Compression, policy
    resolution, primer construction and ledger accounting belong in the codec/policy/stats
    modules and are CALLED by the router.

#237 left open whether to enforce that with a test or leave it as a convention, warning
that "a test that is wrong is worse than a convention that is read". Both candidates it
named were weighed against a real change (PR #246, which added 109 lines of naming logic):

  - A LINE-COUNT CEILING was rejected. #246's growth was routing logic, which is exactly
    what belongs here, so a ceiling would have blocked correct work while catching no
    boundary violation at all. It measures the proxy, not the property.
  - THE IMPORT ASSERTION below was kept. It encodes the rule directly: the router may
    depend on `policy` / `stats` / `lossy` to CALL them, but reaching for the codec itself
    is the signal #237 says should move the decision out rather than widen the file.
"""
from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "terse"

# The codec. `policy`/`stats`/`lossy` are deliberately NOT here: the router calling them is
# the boundary working as intended, not a violation.
CODEC_MODULES = {"transforms"}


def _imports_in_source(source: str, filename: str = "<test>") -> set[str]:
    """Every module name `source` imports, by AST rather than by grep.

    Deliberately not a text search: `multiproxy.py` uses the word "transform" throughout
    its prose (`from_peer` returns a "`pump()`-compatible transform"), so a grep-based
    version of this test would fail on a comment and get deleted as noisy — the exact way
    a wrong test earns less than the convention it replaced.

    Takes SOURCE rather than a path so the detector itself is testable against a known
    violation, without committing a file that violates the boundary to test it with.
    """
    def _top(name: str) -> str:
        # `terse.transforms` and a relative `transforms` are the same module reached two
        # ways, so the absolute spelling has its package stripped before the first
        # component is taken. Without this, `from terse.transforms import compress`
        # reduces to "terse" and sails past the boundary check — found by the
        # known-violation sweep below, which is the whole reason it exists.
        if name.startswith("terse."):
            name = name[len("terse."):]
        return name.split(".")[0]

    tree = ast.parse(source, filename=filename)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(_top(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `from . import x as y` names the MODULE in `names`, not in `module`; both
            # spellings have to count or `from . import transforms` would slip through.
            if node.module:
                found.add(_top(node.module))
            if node.level and not node.module:
                found.update(alias.name for alias in node.names)
    return found


def _imported_modules(path: pathlib.Path) -> set[str]:
    return _imports_in_source(path.read_text(encoding="utf-8"), str(path))


def test_the_router_does_not_reach_into_the_codec():
    # The #237 boundary, stated as an executable property. A failure here is not
    # necessarily a bug — it is the signal that an encoding decision is being made inside
    # the router, and that the decision should move to the codec/policy module and be
    # CALLED from here instead of widening this file.
    imported = _imported_modules(SRC / "multiproxy.py")
    leaked = imported & CODEC_MODULES
    assert not leaked, (
        f"multiproxy.py imports the codec ({', '.join(sorted(leaked))}). Per #237 the "
        "router calls policy/stats/lossy but does not make encoding decisions itself — "
        "move the decision into the codec or policy module and call it from here."
    )


def test_the_boundary_test_can_actually_see_an_import():
    # Guards the guard. `_imported_modules` returning an empty set — a parse that silently
    # found nothing, a refactor to a spelling it does not understand — would make the test
    # above pass vacuously forever. Pin the shape it must keep detecting, including the
    # `from . import x` form that carries the module name in `names` rather than `module`.
    imported = _imported_modules(SRC / "multiproxy.py")
    assert {"policy", "stats", "proxy"} <= imported, imported

    # and it must actually FIRE on each spelling a violation could arrive in
    for violation in ("from . import transforms",
                      "from . import transforms as t",
                      "from .transforms import compress",
                      "from terse.transforms import compress",
                      "import terse.transforms"):
        assert _imports_in_source(violation) & CODEC_MODULES, violation

    # ...without firing on the prose that made a grep-based version unworkable
    assert not _imports_in_source(
        '"""from_peer returns a pump()-compatible transform."""\n') & CODEC_MODULES
