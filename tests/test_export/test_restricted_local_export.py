"""Local export may read restricted datasets; the assisted (MCP) path may not."""
from types import SimpleNamespace

import pytest

from explaneat import governance


def test_local_use_allows_restricted_but_assisted_path_still_refuses():
    ds = SimpleNamespace(name="cohort", additional_metadata={"restricted": True})
    with pytest.raises(governance.GovernanceError):
        governance.require_assisted_access(ds)
    with governance.local_use():
        governance.require_assisted_access(ds)  # no error inside local use
    with pytest.raises(governance.GovernanceError):
        governance.require_assisted_access(ds)  # restored afterwards
