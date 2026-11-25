"""The code of a run is what the command line and a worker can import."""

from bnn.pipeline import provenance


def files(sources) -> set[str]:
    return {path.relative_to(provenance.PACKAGE).as_posix() for path in sources}


def test_run_code_follows_lazy_imports_and_omits_what_no_run_executes():
    code = files(provenance.run_sources())
    # Reached only by imports inside functions or by modules named in literals.
    assert {"pipeline/pipeline.py", "pipeline/worker.py"} <= code
    # Reference targets and views of finished runs never execute in a run.
    assert not code & {
        "target/synthetic.py",
        "samplers/tempered_smc.py",
        "evaluation/cost_error.py",
        "store/trajectories.py",
        "pipeline/samples.py",
    }
    # An extension that a configuration imports becomes code of its run.
    extended = files(provenance.run_sources(("bnn.samplers.tempered_smc",)))
    assert extended - code == {"samplers/tempered_smc.py"}
