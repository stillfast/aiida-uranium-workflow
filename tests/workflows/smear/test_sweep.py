"""Tests for the smear WorkChains' sweep outline (no profile needed).

Both ``AbacusSmearWorkChain`` and ``VaspSmearWorkChain`` read their grid
from backend-natively named ports — ``smearing_method_list`` /
``smearing_sigma_list`` (ABACUS, sigma in Ry) and ``smear_list`` /
``sigma_list`` (VASP, ISMEAR integers + sigma in eV) — and push one
``(method, sigma)`` pair per child by overwriting the child's own
parameters (``smearing_method`` / ``smearing_sigma`` in the ABACUS INPUT
block, ``ismear`` / ``sigma`` in the INCAR).

``submit_children`` is exercised here with a stub ``self`` so the sweep
grid, the child labels and the parameter overrides are pinned without a
scheduler, a transport or a daemon.
"""

from __future__ import annotations

import copy

from aiida import orm
from aiida_uranium_workflow.workflows.smear.abacus import AbacusSmearWorkChain
from aiida_uranium_workflow.workflows.smear.vasp import VaspSmearWorkChain

METHODS = ["mp", "gauss"]
SIGMAS_RY = [0.0225, 0.0150, 0.0100, 0.0075]
ISMEARS = [0, 1]
SIGMAS_EV = [0.3, 0.2, 0.15, 0.1]


class _Bag:
    """Attribute bag that also unpacks like a mapping (``{**bag}``)."""

    def __init__(self, **kwargs):
        self.__dict__["_data"] = dict(kwargs)

    def __getattr__(self, name):
        try:
            return self.__dict__["_data"][name]
        except KeyError as exc:  # mimic a missing namespace entry
            raise AttributeError(name) from exc

    def keys(self):
        return self.__dict__["_data"].keys()

    def __getitem__(self, key):
        return self.__dict__["_data"][key]

    def __contains__(self, key):
        return key in self.__dict__["_data"]

    def get(self, key, default=None):
        return self.__dict__["_data"].get(key, default)


class _StubContext:
    """``self.ctx`` stand-in: attribute get/set plus ``.get``."""

    def __getattr__(self, name):
        try:
            return self.__dict__["_values"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self.__dict__.setdefault("_values", {})[name] = value

    def get(self, name, default=None):
        return self.__dict__.get("_values", {}).get(name, default)


class _StubNode:
    def __init__(self, pk):
        self.pk = pk


def _snapshot(value):
    """Freeze the mutable payload of the inputs handed to a child.

    ``orm.Dict`` / ``orm.List`` nodes that are *not stored yet* (as in this
    stub) alias the very dict/list the outline keeps mutating; the engine
    persists a child's inputs when its parent checkpoints, i.e. after
    ``submit_children`` returned. Freezing the payload at submit time
    therefore reproduces what each child is actually stored with.
    """
    if isinstance(value, orm.Dict):
        return copy.deepcopy(value.get_dict())
    if isinstance(value, orm.List):
        return list(value.get_list())
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value


class _StubWorkChain:
    """Minimal ``WorkChain`` surface used by ``submit_children``."""

    def __init__(self, inputs, exposed=None):
        self.inputs = inputs
        self.ctx = _StubContext()
        self.exposed = exposed or {}
        self.submitted = []
        self.reports = []

    def report(self, message):
        self.reports.append(message)

    def submit(self, workchain_cls, **kwargs):
        self.submitted.append((workchain_cls, _snapshot(kwargs)))
        return _StubNode(pk=len(self.submitted))

    def to_context(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self.ctx, name, value)

    def exposed_inputs(self, workchain_cls, agglomerate=True):
        return _Bag(**self.exposed)

    def _dry_run(self):
        return False


def _abacus_inputs():
    return _Bag(
        abacus=_Bag(
            code=None,
            parameters=orm.Dict(
                {"input": {"calculation": "scf", "smearing_method": "gauss"}}
            ),
            structure=None,
            metadata={},
        ),
        kpoints_distance=orm.Float(0.1),
        pseudo_family=orm.Str("sg15_sz"),
        smearing_method_list=orm.List(METHODS),
        smearing_sigma_list=orm.List(SIGMAS_RY),
    )


def _vasp_inputs():
    exposed = {
        "code": None,
        "structure": None,
        "parameters": orm.Dict({"incar": {"encut": 300}}),
        "potential_family": orm.Str("PBE"),
        "potential_mapping": orm.Dict({"U": "U"}),
        "calc": _Bag(metadata=_Bag(options={"queue_name": "q_test"})),
        "kpoints_spacing": orm.Float(0.0159),
    }
    return _Bag(
        smear_list=orm.List(ISMEARS),
        sigma_list=orm.List(SIGMAS_EV),
        **exposed,
    ), exposed


def test_abacus_sweep_over_method_and_sigma():
    """2 smearing methods × 4 sigmas → 8 children, one per grid point."""
    stub = _StubWorkChain(_abacus_inputs())
    AbacusSmearWorkChain.submit_children(stub)

    assert len(stub.submitted) == len(METHODS) * len(SIGMAS_RY)
    assert stub.ctx.smear_pairs == [
        (method, sigma) for method in METHODS for sigma in SIGMAS_RY
    ]

    seen = set()
    for _, child in stub.submitted:
        abacus = child["abacus"]
        input_block = abacus["parameters"]["input"]
        method = input_block["smearing_method"]
        sigma = input_block["smearing_sigma"]
        seen.add((method, sigma))

        label = child["metadata"]["label"]
        assert method in METHODS and sigma in SIGMAS_RY
        # The child carries the preset's code / structure and the grid point
        # as its parameters — and the sweep ports never leak into it.
        assert "code" in abacus and "structure" in abacus
        assert "smearing_method_list" not in abacus
        assert "smearing_sigma_list" not in child
        assert child["pseudo_family"].value == "sg15_sz"
        assert child["kpoints_distance"].value == 0.1
        # One context entry per grid point, named after it.
        assert getattr(stub.ctx, label).pk > 0

    assert seen == {(method, sigma) for method in METHODS for sigma in SIGMAS_RY}


def test_vasp_sweep_over_ismear_and_sigma():
    """2 ISMEAR values × 4 sigmas → 8 children, each with its own INCAR."""
    inputs, exposed = _vasp_inputs()
    stub = _StubWorkChain(inputs, exposed=exposed)
    VaspSmearWorkChain.submit_children(stub)

    assert len(stub.submitted) == len(ISMEARS) * len(SIGMAS_EV)

    seen = set()
    for _, child in stub.submitted:
        incar = child["parameters"]["incar"]
        seen.add((incar["ismear"], incar["sigma"]))
        # The sweep ports are WorkChain inputs, not child inputs.
        assert "smear_list" not in child and "sigma_list" not in child
        assert child["metadata"]["label"] == (
            f"smear_{incar['ismear']}_sigma_"
            f"{str(incar['sigma']).replace('.', '_')}"
        )
        # ``calc`` (scheduler options) is forwarded to every child.
        assert child["calc"]["metadata"]["options"] == {"queue_name": "q_test"}

    assert seen == {(ismear, sigma) for ismear in ISMEARS for sigma in SIGMAS_EV}
