"""Neutral wording. Reports and explanations must never claim safety or legal compliance.

The rule checker, the repair reports and the explainer use ``CHECKED_PHRASE`` and are linted
with ``find_banned_terms``. The tool only checks constraints that the team configured.
"""

import re

CHECKED_PHRASE = "constraint-checked against the configured rules"
PASS_PHRASE = "passes all configured checks"
FAIL_PHRASE = "does not pass all configured checks"

BANNED_TERMS: tuple[str, ...] = (
    "safe",
    "safely",
    "unsafe",
    "compliant",
    "non-compliant",
    "compliance",
    "approved",
    "certified",
    "guaranteed",
)

_PATTERN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in BANNED_TERMS) + r")(?![A-Za-z])",
    re.IGNORECASE,
)


def find_banned_terms(text: str) -> list[str]:
    """Return banned words found in ``text`` (whole words only; 'safety' is allowed)."""
    return sorted({m.group(1).lower() for m in _PATTERN.finditer(text)})
