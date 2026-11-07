"""Data-governance switches for restricted (e.g. DAP/HDEC-governed) data.

Two levels:

* **Server restricted mode** (``EXPLANEAT_RESTRICTED=1``): the API refuses every
  route that makes an outbound network call (PMLB/UCI catalogue search and
  downloads), and the MCP server refuses all genome and dataset access.
* **Restricted datasets** (``Dataset.additional_metadata["restricted"] is True``):
  the MCP server (the Claude-assisted path, which sends model details and
  evidence to a remote model) refuses genomes trained on them and their splits,
  unless ``EXPLANEAT_ALLOW_ASSISTED_RESTRICTED=1`` is set explicitly.

The API and the React Explorer never send data off the machine themselves:
they bind to 127.0.0.1 by default and the UI bundle loads no external scripts
or fonts.
"""

from __future__ import annotations

import os
from typing import Any, Optional

_TRUE = {"1", "true", "yes", "on"}


class GovernanceError(PermissionError):
    """An action is disabled by the data-governance settings."""


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def restricted_server() -> bool:
    return _flag("EXPLANEAT_RESTRICTED")


def assisted_restricted_allowed() -> bool:
    return _flag("EXPLANEAT_ALLOW_ASSISTED_RESTRICTED")


def dataset_is_restricted(dataset: Any) -> bool:
    meta = getattr(dataset, "additional_metadata", None) or {}
    return bool(isinstance(meta, dict) and meta.get("restricted") is True)


def status() -> dict:
    return {
        "restricted_server": restricted_server(),
        "assisted_restricted_allowed": assisted_restricted_allowed(),
        "external_network_allowed": not restricted_server(),
    }


def require_external_network(what: str = "this action") -> None:
    if restricted_server():
        raise GovernanceError(
            f"{what} needs an outbound network connection, which is disabled in "
            "restricted mode (EXPLANEAT_RESTRICTED=1)"
        )


def require_assisted_access(dataset: Optional[Any]) -> None:
    """Raise unless the Claude/MCP-assisted path may read this dataset."""
    if restricted_server() and not assisted_restricted_allowed():
        raise GovernanceError(
            "The MCP (Claude-assisted) path is disabled in restricted mode "
            "(EXPLANEAT_RESTRICTED=1). Set EXPLANEAT_ALLOW_ASSISTED_RESTRICTED=1 "
            "only if the data's governance permits it."
        )
    if dataset is not None and dataset_is_restricted(dataset) and not assisted_restricted_allowed():
        raise GovernanceError(
            f"Dataset {getattr(dataset, 'name', '')!r} is marked restricted: the MCP "
            "(Claude-assisted) path is disabled for it. Explain it in the local UI, or "
            "set EXPLANEAT_ALLOW_ASSISTED_RESTRICTED=1 only if its governance permits."
        )
