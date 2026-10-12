"""Shared setup for the acorn pitfall tests (brief §4.5).

Default: import the installed (pinned + patches/acorn) acorn; if it is not installed yet, fall back to the patched
clone at $HNL_ENV/src/acorn. With ACORN_PIN_SRC=/work/submit/jinboz1/hnl/tmp/acorn_pin_src the UNPATCHED pinned
source is imported instead, to show the behaviour before the patches (tests of patched pitfalls are then expected to
fail). The `acorn_origin` fixture reports which copy was imported.
"""

import os
import sys

_pin = os.environ.get("ACORN_PIN_SRC")
if _pin:
    sys.path.insert(0, _pin)
else:
    try:
        import acorn  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(os.environ.get("HNL_ENV", "/work/submit/jinboz1/hnl/env"), "src", "acorn"))

import acorn  # noqa: E402
import pytest  # noqa: E402

if _pin and not os.path.realpath(acorn.__file__).startswith(os.path.realpath(_pin)):
    raise RuntimeError(f"ACORN_PIN_SRC set but acorn was imported from {acorn.__file__}")


@pytest.fixture(scope="session")
def acorn_origin():
    return "pin" if _pin else "patched"
