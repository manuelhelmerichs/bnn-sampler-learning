# Saved scientific results

`load_record(directory)` reads a completed result of any profile, including
rejected scientific outcomes. It returns `None` when no summary exists and
rejects inconsistent or incomplete saved records. It does not load experiment
definitions or compare the producing code with today's checkout.

Each result directory contains:

- `manifest.json`: experiment, profile, question, factors, seeds, input identities,
  and measurement-code provenance.
- `cells/<cell_id>/result.json` and `record.json`: measured values and execution
  completion. Named arrays or other scientific artifacts stay beside the cell.
- `summary.json`: consumed cell IDs, original cell
  metrics, reducer identity, study aggregates, qualification and eligibility
  flags, and study-specific units and uncertainty assumptions.
- Referenced geometry files and producing code snapshots, where present.

Readers validate required fields, cell membership, and artifacts directly.
Refresh saved reductions when a consumer needs statistics that are missing.

`Record.rows` joins manifest factors, measured metrics, and saved
`row_summaries[cell_id]`. Derived statistics cannot overwrite measurements.
`part_record(record, name)` selects a joined experiment's exact saved membership
and part metadata. Original cell metrics remain available separately.

`Record.artifact(cell_id, relative_path, sha256)` resolves a cell-local artifact
and optionally validates its bytes. Geometry references are result-relative and
validated when loaded. Copy the whole result directory to preserve its reader
contract; absolute acquisition paths remain provenance, not consumer inputs.

Statistical units and assumptions are study-specific: retain resampling units,
seeds, interval levels, descriptive-range labels, complete failure denominators,
and any fitted-coordinate selection metadata alongside the quantities. Website
eligibility is an application policy, separate from this general reader.
