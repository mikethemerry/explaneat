"""Rename every input node of a genome's phenotype to its feature name (rename_node ops).

Input node -k is feature column k-1 (neat-python convention). Run once, before any split.

    uv run python experiments/10seed/rename_inputs.py <genome_id> <dataset_id>
"""

import json
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp_server.helpers import build_model_state  # noqa: E402
from mcp_server.server import get_db  # noqa: E402
from mcp_server.tools.operations import apply_operation  # noqa: E402
from explaneat.db.models import Dataset  # noqa: E402


def main(genome_id, dataset_id):
    db = get_db()
    with db.session_scope() as session:
        names = list(session.get(Dataset, UUID(dataset_id)).feature_names)
        inputs = [str(n) for n in build_model_state(session, genome_id).input_node_ids]
    for nid in sorted(inputs, key=lambda s: -int(s)):
        name = names[-int(nid) - 1]
        r = json.loads(apply_operation(genome_id, "rename_node",
                                       json.dumps({"node_id": nid, "display_name": name})))
        print(nid, "->", name, "" if "error" not in r else f"ERROR {r['error']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
