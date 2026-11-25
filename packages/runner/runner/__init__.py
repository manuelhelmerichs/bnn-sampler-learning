"""Run the experiments of the repository and keep their results.

An experiment declares its cells, measures one at a time and summarizes them.
The runner plans the cells, runs them within a resource budget, records which
commit and which ``bnn`` runs produced every result, and reads results back.
"""

from runner.artifacts import atomic_json, atomic_npz, read_arrays, read_json
from runner.cells import CellSpec, Design, ResourceHint
from runner.experiment import Experiment, bind_prepared, reduce_protocol
from runner.lifecycle import cell_folder, finite, result_folder
from runner.paths import relative, repository_root
from runner.provenance import checkout, runtime_provenance
from runner.results import Record, load_record, part_record
from runner.seeds import derive_seed
from runner.serialization import fingerprint

__all__ = [
    "CellSpec",
    "Design",
    "Experiment",
    "Record",
    "ResourceHint",
    "atomic_json",
    "atomic_npz",
    "bind_prepared",
    "cell_folder",
    "checkout",
    "derive_seed",
    "finite",
    "fingerprint",
    "load_record",
    "part_record",
    "read_arrays",
    "read_json",
    "reduce_protocol",
    "relative",
    "repository_root",
    "result_folder",
    "runtime_provenance",
]
