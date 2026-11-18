"""Split-node id conventions: ``<base>_<suffix>``.

A split copy of node ``5`` is ``5_<token>``; a consolidated node joins its copies' tokens in
canonical order (``5_a`` + ``5_c`` -> ``5_ac``). The suffix is therefore a *sequence of tokens*:

* tokens 0-25 are the single lowercase letters ``a`` .. ``z`` (the original scheme, unchanged, so
  every stored explanation replays to the same ids);
* tokens 26 and up are two characters, an uppercase letter then a lowercase letter:
  ``Aa`` .. ``Az``, ``Ba`` .. ``Zz`` (676 more, 702 in all).

A multi-character lowercase suffix (``5_aa``) cannot mean "copy 27": it already means "the
consolidation of copies a and a", so lowercase pairs are never used as single tokens. The
uppercase lead character keeps the token sequence prefix-free: ``5_aAbc`` is the consolidation of
``a``, ``Ab`` and ``c``. Every suffix is alphabetic, so ``str.isalpha`` tests keep working.
"""

from __future__ import annotations

import re
from typing import List, Optional

LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = LOWER.upper()
MAX_TOKENS = 26 + 26 * 26

#: one token: a lowercase letter, or an uppercase letter followed by a lowercase letter
TOKEN_RE = r"(?:[a-z]|[A-Z][a-z])"
#: a whole suffix (one or more tokens)
SUFFIX_RE = TOKEN_RE + "+"
_SUFFIX = re.compile(rf"^{SUFFIX_RE}$")
_TOKEN = re.compile(TOKEN_RE)
_SPLIT_ID = re.compile(rf"^(.+)_({SUFFIX_RE})$")


def token(i: int) -> str:
    """The i-th copy token (0-based): a..z, then Aa..Az, Ba..Bz, ... Zz."""
    if i < 0 or i >= MAX_TOKENS:
        raise ValueError(f"split token index {i} out of range (max {MAX_TOKENS - 1})")
    if i < 26:
        return LOWER[i]
    j = i - 26
    return UPPER[j // 26] + LOWER[j % 26]


def token_index(tok: str) -> int:
    if len(tok) == 1 and tok in LOWER:
        return LOWER.index(tok)
    if len(tok) == 2 and tok[0] in UPPER and tok[1] in LOWER:
        return 26 + 26 * UPPER.index(tok[0]) + LOWER.index(tok[1])
    raise ValueError(f"not a split token: {tok!r}")


def is_suffix(s: str) -> bool:
    return bool(s) and bool(_SUFFIX.match(s))


def tokens(suffix: str) -> List[str]:
    """``"aAbc"`` -> ``["a", "Ab", "c"]``. Raises ValueError for a malformed suffix."""
    if not is_suffix(suffix):
        raise ValueError(f"not a split suffix: {suffix!r}")
    return _TOKEN.findall(suffix)


def join(toks) -> str:
    """Canonical consolidated suffix: tokens in index order (``a`` < ``z`` < ``Aa`` < ``Zz``).
    For lowercase-only tokens this is the old ``"".join(sorted(...))``."""
    return "".join(sorted(toks, key=token_index))


def split_id(node_id: str):
    """``"5_aAb"`` -> ``("5", "aAb")``; ``(node_id, None)`` when it is not a split id."""
    m = _SPLIT_ID.match(str(node_id))
    if not m:
        return str(node_id), None
    return m.group(1), m.group(2)


def base_id(node_id: str) -> Optional[str]:
    """Base of a split id (``"5_Ab"`` -> ``"5"``); the id itself when it has no ``_``; None when
    it has an ``_`` but is not a split id (e.g. ``identity_5``)."""
    node_id = str(node_id)
    if "_" not in node_id:
        return node_id
    b, s = split_id(node_id)
    return b if s else None


def suffix(node_id: str) -> Optional[str]:
    return split_id(node_id)[1]


def strip(node_id: str) -> str:
    """Base id when ``node_id`` is a split id, else ``node_id`` unchanged."""
    return split_id(node_id)[0]
