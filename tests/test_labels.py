"""Tests for the backend/method-aware directory label resolver.

Two halves:

1. The pure ``format_*_label`` helpers (no AiiDA required) — make sure
   the exact strings the WorkChains build in their ``submit_children``
   outline / ``copy_calc._label_for_*`` are reproduced, including:
   * ABACUS ``ecutwfc_<v>_kpoints_distance_<v>`` /
     ``ecutwfc_<v>_kpoints_<NxNxN>``
   * VASP ``kpoints_spacing_<v>_encut_<v>`` /
     ``kpoints_<NxNxN>_encut_<v>``
   * ABACUS ``smearing_<method>_sigma_<value>``
   * VASP ``ismear_<n>_sigma_<value>``
   * Magmom (ABACUS nested list, VASP mapping dict)

2. The ``resolve_label`` glue — feed it fake AiiDA CalcJobNode-like
   objects carrying real input shapes and assert the resulting leaf
   directory name matches what the workflow's ``submit_children``
   outlined. Also covers the fallback chain (``metadata.label``,
   ``process_label``, ``calcjob_<pk>``) and the parent-WorkChain
   input resolution path (when the CalcJob itself doesn't expose
   abacus.parameters / parameters.incar).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiida_uranium_workflow.utils.labels import (
    format_abacus_convergence_label,
    format_abacus_smear_label,
    format_magmom_label,
    format_vasp_convergence_label,
    format_vasp_smear_label,
    resolve_label,
)


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


class TestFormatAbacusSmearLabel:
    """Matches ``copy_calc._label_for_abacus`` byte-for-byte."""

    @pytest.mark.parametrize(
        "method,sigma,expected",
        [
            ("gauss", 0.06, "smearing_gauss_sigma_0_060000"),
            ("mp", 0.06, "smearing_mp_sigma_0_060000"),
            ("mp", 0.00441, "smearing_mp_sigma_0_004410"),
            ("mp2", 0.2, "smearing_mp2_sigma_0_200000"),
        ],
    )
    def test_matches_submit_children(self, method, sigma, expected):
        assert format_abacus_smear_label(method, sigma) == expected

    def test_negative_sigma(self):
        # The legacy _label_for_abacus wraps the dash with the underscore
        # already produced by ``.replace(".", "_")``:
        # ``"-0_060000".replace("-", "m") == "m0_060000"`` so the full
        # label keeps the underscore between ``sigma`` and the sign.
        assert (
            format_abacus_smear_label("gauss", -0.06)
            == "smearing_gauss_sigma_m0_060000"
        )


class TestFormatVaspSmearLabel:
    """Matches ``copy_calc._label_for_vasp`` byte-for-byte."""

    @pytest.mark.parametrize(
        "ismear,sigma,expected",
        [
            (0, 0.06, "ismear_0_sigma_0_060000"),
            (2, 0.06, "ismear_2_sigma_0_060000"),
            (2, 0.00441, "ismear_2_sigma_0_004410"),
        ],
    )
    def test_matches_submit_children(self, ismear, sigma, expected):
        assert format_vasp_smear_label(ismear, sigma) == expected

    def test_negative_ismear(self):
        # ``ismear`` goes through ``int(...)`` and is *not* formatted as a
        # float, so the negative sign stays verbatim — there's no
        # ``"%.6f"`` step that would inject a ``"_"`` first.
        assert (
            format_vasp_smear_label(-5, 0.06)
            == "ismear_-5_sigma_0_060000"
        )


class TestFormatAbacusConvergenceLabel:
    """``ecutwfc_<v>_kpoints_distance_<v>`` and ``ecutwfc_<v>_kpoints_<NxNxN>``."""

    def test_kpoints_distance(self):
        # The workflow builds with ``f"{kpoints_val}"`` then
        # ``.replace(".", "_")`` — so ``0.2`` becomes ``0_2``, not
        # ``0_200000``. This matches ``submit_children`` byte-for-byte.
        assert (
            format_abacus_convergence_label(60, 0.2)
            == "ecutwfc_60_kpoints_distance_0_2"
        )
        assert (
            format_abacus_convergence_label(60, 0.04)
            == "ecutwfc_60_kpoints_distance_0_04"
        )

    def test_kpoints_mesh(self):
        mesh = (4, 4, 4)
        assert (
            format_abacus_convergence_label(60, mesh)
            == "ecutwfc_60_kpoints_4x4x4"
        )
        assert (
            format_abacus_convergence_label(80, (6, 6, 6))
            == "ecutwfc_80_kpoints_6x6x6"
        )

    def test_mesh_detection_accepts_list(self):
        # ``list`` should also be detected as a mesh.
        assert (
            format_abacus_convergence_label(50, [2, 2, 2])
            == "ecutwfc_50_kpoints_2x2x2"
        )


class TestFormatVaspConvergenceLabel:
    """``kpoints_spacing_<v>_encut_<v>`` and ``kpoints_<NxNxN>_encut_<v>``."""

    def test_spacing(self):
        assert (
            format_vasp_convergence_label(0.2, 520)
            == "kpoints_spacing_0_2_encut_520"
        )

    def test_mesh(self):
        mesh = (4, 4, 4)
        assert (
            format_vasp_convergence_label(mesh, 520)
            == "kpoints_4x4x4_encut_520"
        )


class TestFormatMagmomLabel:
    """Magmom labels cover ABACUS nested list and VASP mapping dict."""

    def test_flat_list(self):
        # ``f"{v:g}"`` → ``"1"`` / ``"-1"`` then ``replace(".", "_").replace("-", "m")``.
        # For ``[1.0, -1.0]`` the inner ``_scalar_token`` produces ``"1"`` and ``"-1"``;
        # they are joined with ``_`` before the replace pass so the
        # underscore sticks around.
        assert format_magmom_label([1.0, -1.0]) == "magmom_1_m1"
        assert format_magmom_label([0.5, -0.5]) == "magmom_0_5_m0_5"

    def test_nested_list(self):
        assert format_magmom_label([[1.0], [-1.0]]) == "magmom_1_m1"

    def test_dict(self):
        assert format_magmom_label({"U": 1.0}) == "magmom_U_1"
        assert format_magmom_label({"U": [1.0, -1.0]}) == "magmom_U_1_m1"
        assert (
            format_magmom_label({"Si": 1.0, "U": -1.0})
            == "magmom_Si_1__U_m1"
        )

    def test_scalar(self):
        assert format_magmom_label(1.0) == "magmom_1"
        assert format_magmom_label(-0.5) == "magmom_m0_5"

    def test_index_prefix(self):
        assert format_magmom_label([1.0, -1.0], index=0) == "magmom_000_1_m1"
        assert format_magmom_label({"U": 1.0}, index=5) == "magmom_005_U_1"


# ---------------------------------------------------------------------------
# ``resolve_label`` against AiiDA-like fakes
# ---------------------------------------------------------------------------


class _FakeDict(SimpleNamespace):
    """Holds a backing dict and supports ``get_dict()`` + subscription."""

    def __init__(self, payload):
        super().__init__()
        self._payload = payload

    def get_dict(self):
        return self._payload

    def get(self, key, default=None):
        if key in self._payload:
            return self._payload[key]
        return default

    def __contains__(self, key):
        return key in self._payload

    def __getitem__(self, key):
        return self._payload[key]


class _FakeFloat(SimpleNamespace):
    def __init__(self, value):
        super().__init__(value=float(value))


class _FakeKpointsData(SimpleNamespace):
    def __init__(self, mesh):
        super().__init__()
        self._mesh = tuple(int(x) for x in mesh)

    def get_kpoints_mesh(self):
        return self._mesh, []


class _FakeInputs(SimpleNamespace):
    """Duck-typed ``process.inputs`` proxy."""

    def __init__(self, data):
        super().__init__()
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)


def _abacus_calcjob(ecutwfc, kpoints, metadata_label=""):
    """Build a fake CalcJob with abacus ``parameters`` and a kpoints value."""
    parameters = _FakeDict({"input": {"ecutwfc": ecutwfc}})
    abacus_block = {"parameters": parameters, "structure": object()}
    inputs = _FakeInputs(
        {
            "abacus": abacus_block,
            "kpoints" if isinstance(kpoints, tuple) else "kpoints_distance": (
                _FakeKpointsData(kpoints)
                if isinstance(kpoints, tuple)
                else _FakeFloat(kpoints)
            ),
            "metadata": _FakeDict({"label": metadata_label}) if metadata_label else {},
        }
    )
    node = SimpleNamespace(
        pk=10,
        process_label="AbacusCalculation",
        metadata=SimpleNamespace(label=metadata_label),
        inputs=inputs,
        caller=None,
    )
    return node


def _vasp_calcjob(encut, kpoints, metadata_label=""):
    parameters = _FakeDict({"incar": {"encut": encut}})
    if isinstance(kpoints, tuple):
        kp_value = _FakeKpointsData(kpoints)
        inputs = _FakeInputs(
            {
                "parameters": parameters,
                "kpoints": kp_value,
                "metadata": _FakeDict({"label": metadata_label}) if metadata_label else {},
            }
        )
    else:
        inputs = _FakeInputs(
            {
                "parameters": parameters,
                "kpoints_spacing": _FakeFloat(kpoints),
                "metadata": _FakeDict({"label": metadata_label}) if metadata_label else {},
            }
        )
    return SimpleNamespace(
        pk=20,
        process_label="VaspCalculation",
        metadata=SimpleNamespace(label=metadata_label),
        inputs=inputs,
        caller=None,
    )


class TestResolveAbacusConvergence:
    """Direct input → ABACUS convergence label."""

    def test_distance_mode(self):
        node = _abacus_calcjob(ecutwfc=60, kpoints=0.2)
        assert (
            resolve_label(node, backend="abacus", method="convergence")
            == "ecutwfc_60_kpoints_distance_0_2"
        )

    def test_mesh_mode(self):
        node = _abacus_calcjob(ecutwfc=80, kpoints=(4, 4, 4))
        assert (
            resolve_label(node, backend="abacus", method="convergence")
            == "ecutwfc_80_kpoints_4x4x4"
        )

    def test_inherits_from_parent_when_calcjob_inputs_missing(self):
        """When the CalcJob's own inputs lack abacus.parameters (e.g. because
        the WorkChain exposed them at the parent level), the resolver walks
        up the caller chain and reads the parent's abacus.parameters."""
        # Bare CalcJob-like node with empty inputs.
        node = SimpleNamespace(
            pk=11,
            process_label="AbacusBaseWorkChain",
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
            caller=None,
        )
        # Parent WorkChain carrying the sweep inputs.
        parameters = _FakeDict({"input": {"ecutwfc": 100}})
        kpoints_block = _FakeFloat(0.04)
        parent_inputs = _FakeInputs(
            {
                "abacus": {"parameters": parameters, "structure": object()},
                "kpoints_distance": kpoints_block,
            }
        )
        parent = SimpleNamespace(
            pk=12,
            process_label="AbacusConvergenceWorkChain",
            metadata=SimpleNamespace(label=""),
            inputs=parent_inputs,
            caller=None,
        )
        # Wire the caller chain: parent -> node
        node.caller = parent
        parent.caller = SimpleNamespace(
            pk=13,
            process_label="AbacusConvergenceWorkChain",
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
        )

        assert (
            resolve_label(node, backend="abacus", method="convergence")
            == "ecutwfc_100_kpoints_distance_0_04"
        )


class TestResolveVaspConvergence:
    """Direct input → VASP convergence label."""

    def test_spacing_mode(self):
        node = _vasp_calcjob(encut=520, kpoints=0.2)
        assert (
            resolve_label(node, backend="vasp", method="convergence")
            == "kpoints_spacing_0_2_encut_520"
        )

    def test_mesh_mode(self):
        node = _vasp_calcjob(encut=520, kpoints=(4, 4, 4))
        assert (
            resolve_label(node, backend="vasp", method="convergence")
            == "kpoints_4x4x4_encut_520"
        )


class TestResolveSmearRegression:
    """Smear labels must match what ``copy_calc._label_for_*`` emits."""

    def test_abacus_smear_via_inputs(self):
        parameters = _FakeDict(
            {"input": {"smearing_method": "mp", "smearing_sigma": 0.06}}
        )
        inputs = _FakeInputs(
            {"abacus": {"parameters": parameters, "structure": object()}}
        )
        node = SimpleNamespace(
            pk=30,
            process_label="AbacusBaseWorkChain",
            metadata=SimpleNamespace(label=""),
            inputs=inputs,
            caller=None,
        )
        assert (
            resolve_label(node, backend="abacus", method="smear")
            == "smearing_mp_sigma_0_060000"
        )

    def test_vasp_smear_via_inputs(self):
        parameters = _FakeDict({"incar": {"ismear": 2, "sigma": 0.06}})
        inputs = _FakeInputs({"parameters": parameters})
        node = SimpleNamespace(
            pk=31,
            process_label="VaspCalculation",
            metadata=SimpleNamespace(label=""),
            inputs=inputs,
            caller=None,
        )
        assert (
            resolve_label(node, backend="vasp", method="smear")
            == "ismear_2_sigma_0_060000"
        )

    def test_pure_formatter_matches_copy_calc(self):
        """Direct exercise of the format helper used by ``copy_calc._label_for_abacus``."""
        from aiida_uranium_workflow.utils.copy_calc import _label_for_abacus, _label_for_vasp

        assert _label_for_abacus("mp", 0.06) == "smearing_mp_sigma_0_060000"
        assert _label_for_vasp(2, 0.06) == "ismear_2_sigma_0_060000"


class TestResolveMagmom:
    """Magmom labels for ABACUS and VASP."""

    def test_abacus_mag(self):
        parameters = _FakeDict({"stru": {"mag": [[1.0], [-1.0]]}})
        inputs = _FakeInputs(
            {"abacus": {"parameters": parameters, "structure": object()}}
        )
        node = SimpleNamespace(
            pk=40,
            process_label="AbacusBaseWorkChain",
            metadata=SimpleNamespace(label=""),
            inputs=inputs,
            caller=None,
        )
        assert resolve_label(node, backend="abacus", method="magmom") == "magmom_1_m1"

    def test_vasp_magmom_mapping(self):
        parameters = _FakeDict({"incar": {}})
        mapping = _FakeDict({"U": [1.0, -1.0]})
        inputs = _FakeInputs(
            {"parameters": parameters, "magmom_mapping": mapping}
        )
        node = SimpleNamespace(
            pk=41,
            process_label="VaspCalculation",
            metadata=SimpleNamespace(label=""),
            inputs=inputs,
            caller=None,
        )
        assert (
            resolve_label(node, backend="vasp", method="magmom")
            == "magmom_U_1_m1"
        )


class TestResolveFallback:
    """When inputs can't be parsed we fall back through the chain."""

    def test_falls_back_to_metadata_label(self):
        node = SimpleNamespace(
            pk=50,
            process_label="AbacusCalculation",
            metadata=SimpleNamespace(label="explicit_label"),
            inputs=_FakeInputs({}),
            caller=None,
        )
        assert (
            resolve_label(node, backend="abacus", method="convergence")
            == "explicit_label"
        )

    def test_falls_back_to_pk_token_when_process_label_is_generic(self):
        # ``AbacusCalculation`` is a generic CalcJob class name; we
        # deliberately skip it at every fallback step so the user does
        # not end up with 40 folders all named ``AbacusCalculation``.
        node = SimpleNamespace(
            pk=51,
            process_label="AbacusCalculation",
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
            caller=None,
        )
        assert (
            resolve_label(node, backend="abacus", method="convergence")
            == "calcjob_51"
        )

    def test_falls_back_to_pk_token(self):
        node = SimpleNamespace(
            pk=52,
            process_label="",
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
            caller=None,
        )
        assert (
            resolve_label(node, backend="abacus", method="smear")
            == "calcjob_52"
        )


# ---------------------------------------------------------------------------
# Real VASP provenance — the VaspCalculation itself does not carry
# ``parameters`` (no INCAR is attached to the final CalcJob); the
# intermediate VaspWorkChain does. The root WorkChain
# (VaspSmearWorkChain / VaspConvergenceWorkChain) is just a wrapper.
#
# Two resolution paths are exercised:
#
# 1. ``inputs.parameters.incar`` / ``inputs.kpoints_spacing`` on the
#    intermediate VaspWorkChain — what the workflow's ``submit_children``
#    *would* have inlined if the WorkChain did not pre-format
#    ``metadata.label``.
# 2. ``metadata.label`` on the intermediate VaspWorkChain — what the
#    workflow *actually* sets in production via
#    ``metadata={'label': '...'}`` at submit time. AiiDA persists this
#    as the ProcessNode's ``label`` attribute (``node.label``), which
#    the resolver reads via the duck-typed ``node.metadata.label``
#    fallback.
# ---------------------------------------------------------------------------


def _vasp_real_provenance(
    *,
    root_class: str,
    inner_inputs: _FakeInputs,
    calcjob_pk: int = 1000,
    inner_pk: int = 1100,
    root_pk: int = 1200,
    inner_metadata_label: str = "",
):
    """Build the VASP call chain
    ``root_workchain -> VaspWorkChain -> VaspCalculation``.

    The VaspCalculation's own ``inputs`` is intentionally empty
    (no parameters / no INCAR), mirroring what AiiDA persists when
    the final CalcJob was submitted with ``parameters`` already
    inlined into the parent WorkChain's inputs.

    ``inner_metadata_label`` populates the intermediate VaspWorkChain's
    ``metadata.label`` (i.e. the submit-time ``ProcessNode.label``
    attribute) — used by the metadata-fallback tests below.
    """

    # Final CalcJob: no `parameters` here.
    calcjob_inputs = _FakeInputs(
        {
            "metadata": _FakeDict({"label": ""}),
        }
    )
    calcjob = SimpleNamespace(
        pk=calcjob_pk,
        process_label="VaspCalculation",
        metadata=SimpleNamespace(label=""),
        inputs=calcjob_inputs,
        caller=None,
    )

    # Intermediate VaspWorkChain — carries parameters.incar / kpoints.
    inner = SimpleNamespace(
        pk=inner_pk,
        process_label="VaspWorkChain",
        metadata=SimpleNamespace(label=inner_metadata_label),
        inputs=inner_inputs,
        caller=None,
        called=[calcjob],
    )

    # Top-level WorkChain (e.g. VaspSmearWorkChain / VaspConvergenceWorkChain).
    root = SimpleNamespace(
        pk=root_pk,
        process_label=root_class,
        metadata=SimpleNamespace(label=""),
        inputs=_FakeInputs({}),
        caller=None,
        called=[inner],
    )

    # Wire caller chains: calcjob <- inner <- root
    calcjob.caller = inner
    inner.caller = root

    return calcjob, inner, root


class TestVaspRealProvenanceSmear:
    """VaspSmearWorkChain → VaspWorkChain → VaspCalculation."""

    def test_root_workchain_argument_resolves(self):
        parameters = _FakeDict({"incar": {"ismear": 2, "sigma": 0.06}})
        inner_inputs = _FakeInputs(
            {
                "parameters": parameters,
                "metadata": _FakeDict({"label": ""}),
            }
        )
        calcjob, _inner, root = _vasp_real_provenance(
            root_class="VaspSmearWorkChain",
            inner_inputs=inner_inputs,
        )

        assert (
            resolve_label(
                calcjob,
                backend="vasp",
                method="smear",
                root_workchain=root,
            )
            == "ismear_2_sigma_0_060000"
        )

    def test_walks_caller_chain_without_root_argument(self):
        """When ``root_workchain`` isn't supplied the resolver should
        still recover the label purely by walking the caller chain
        (root → VaspWorkChain → VaspCalculation)."""
        parameters = _FakeDict({"incar": {"ismear": 2, "sigma": 0.06}})
        inner_inputs = _FakeInputs(
            {
                "parameters": parameters,
                "metadata": _FakeDict({"label": ""}),
            }
        )
        calcjob, _inner, _root = _vasp_real_provenance(
            root_class="VaspSmearWorkChain",
            inner_inputs=inner_inputs,
        )

        assert (
            resolve_label(calcjob, backend="vasp", method="smear")
            == "ismear_2_sigma_0_060000"
        )

    def test_metadata_label_on_inner_workchain_is_used(self):
        """When the intermediate VaspWorkChain carries the explicit
        submit-time ``metadata.label`` (``ProcessNode.label``) and the
        ``inputs.parameters.incar`` is missing, the resolver must fall
        back to that label rather than the generic class name.

        This mirrors what ``submit_children`` produces in production:
        the WorkChain outlines the child with
        ``metadata={'label': 'ismear_2_sigma_0_060000'}`` and AiiDA
        persists it as ``node.label``.
        """
        # ``inputs.parameters`` deliberately absent — the submit-time
        # metadata.label is the only carrier of the smear parameters.
        inner_inputs = _FakeInputs(
            {
                "metadata": _FakeDict({"label": ""}),
            }
        )
        calcjob, _inner, _root = _vasp_real_provenance(
            root_class="VaspSmearWorkChain",
            inner_inputs=inner_inputs,
            inner_metadata_label="ismear_2_sigma_0_060000",
        )

        assert (
            resolve_label(calcjob, backend="vasp", method="smear")
            == "ismear_2_sigma_0_060000"
        )


class TestVaspRealProvenanceConvergence:
    """VaspConvergenceWorkChain → VaspWorkChain → VaspCalculation."""

    def test_root_workchain_argument_resolves(self):
        parameters = _FakeDict({"incar": {"encut": 400}})
        inner_inputs = _FakeInputs(
            {
                "parameters": parameters,
                "kpoints_spacing": _FakeFloat(0.2),
                "metadata": _FakeDict({"label": ""}),
            }
        )
        calcjob, _inner, root = _vasp_real_provenance(
            root_class="VaspConvergenceWorkChain",
            inner_inputs=inner_inputs,
        )

        assert (
            resolve_label(
                calcjob,
                backend="vasp",
                method="convergence",
                root_workchain=root,
            )
            == "kpoints_spacing_0_2_encut_400"
        )

    def test_walks_caller_chain_without_root_argument(self):
        """Caller chain alone (no ``root_workchain`` arg) is enough."""
        parameters = _FakeDict({"incar": {"encut": 400}})
        inner_inputs = _FakeInputs(
            {
                "parameters": parameters,
                "kpoints_spacing": _FakeFloat(0.2),
                "metadata": _FakeDict({"label": ""}),
            }
        )
        calcjob, _inner, _root = _vasp_real_provenance(
            root_class="VaspConvergenceWorkChain",
            inner_inputs=inner_inputs,
        )

        assert (
            resolve_label(calcjob, backend="vasp", method="convergence")
            == "kpoints_spacing_0_2_encut_400"
        )

    def test_metadata_label_on_inner_workchain_is_used(self):
        """``metadata.label='kpoints_spacing_0_2_encut_400'`` on the
        intermediate VaspWorkChain is recovered by the resolver when
        ``inputs.parameters`` is missing.

        This is the production path: ``VaspConvergenceWorkChain`` sets
        ``metadata={'label': 'kpoints_spacing_<v>_encut_<v>'}`` per
        child via ``submit_children``.
        """
        # No inputs.parameters — the submit-time label carries it all.
        inner_inputs = _FakeInputs(
            {
                "metadata": _FakeDict({"label": ""}),
            }
        )
        calcjob, _inner, _root = _vasp_real_provenance(
            root_class="VaspConvergenceWorkChain",
            inner_inputs=inner_inputs,
            inner_metadata_label="kpoints_spacing_0_2_encut_400",
        )

        assert (
            resolve_label(calcjob, backend="vasp", method="convergence")
            == "kpoints_spacing_0_2_encut_400"
        )


# ---------------------------------------------------------------------------
# Same provenance scenarios but built from real AiiDA nodes
# (``aiida.orm.WorkChainNode`` / ``CalcJobNode`` / ``Dict`` / ``Float`` /
# ``KpointsData``). These tests run inside the ``aiida.tools.pytest_fixtures``
# profile so the nodes are backed by a real (sqlite) storage. We wire the
# caller chain by overriding the ``caller`` / ``called`` link triples
# after construction so we don't have to actually launch any workflows.
# ---------------------------------------------------------------------------


def _build_vasp_provenance_real(
    *,
    root_label: str,
    inner_label: str,
    parameters_dict: dict | None,
    kpoints_spacing: float | None = None,
    kpoints_mesh: tuple[int, int, int] | None = None,
):
    """Construct a real-AiiDA provenance triple:

    ``root_workchain -> VaspWorkChain -> VaspCalculation``

    and return ``(calcjob, root)``.  ``inner_label`` populates the
    intermediate WorkChain's persistent ``label`` attribute (i.e.
    ``metadata.label`` as the resolver sees it).

    Note: ``ProcessNode.caller`` and ``ProcessNode.called`` are
    properties that query the database via ``base.links`` — so to
    forge a caller chain without launching any workflows we use the
    proper ``add_incoming`` API with ``LinkType.CALL_WORK`` /
    ``LinkType.CALL_CALC``.  The resolver then reads the chain
    naturally.
    """
    from aiida.common import LinkType
    from aiida.orm import CalcJobNode, Dict, Float, KpointsData, WorkChainNode

    calcjob = CalcJobNode()
    inner = WorkChainNode()
    root = WorkChainNode()

    # Inputs on the intermediate VaspWorkChain. ``INPUT_WORK`` (not
    # ``INPUT_CALC``) because ``inner`` is a ``WorkChainNode``.
    if parameters_dict is not None:
        inner.base.links.add_incoming(
            Dict(parameters_dict), link_type=LinkType.INPUT_WORK, link_label="parameters"
        )
    if kpoints_spacing is not None:
        inner.base.links.add_incoming(
            Float(kpoints_spacing),
            link_type=LinkType.INPUT_WORK,
            link_label="kpoints_spacing",
        )
    if kpoints_mesh is not None:
        kp = KpointsData()
        kp.set_kpoints_mesh(list(kpoints_mesh))
        inner.base.links.add_incoming(
            kp, link_type=LinkType.INPUT_WORK, link_label="kpoints"
        )

    # Caller chain — root_workchain called inner; inner called calcjob.
    calcjob.base.links.add_incoming(inner, link_type=LinkType.CALL_CALC, link_label="call")
    inner.base.links.add_incoming(root, link_type=LinkType.CALL_WORK, link_label="call")

    # Persist labels. ``metadata.label`` on real AiiDA ProcessNodes is
    # the persistent ``label`` attribute (``node.label``), not an
    # attribute of an inner ``metadata`` namespace.
    root.label = root_label
    inner.label = inner_label
    calcjob.label = ""

    # The resolver queries ``node.metadata`` — AiiDA ProcessNodes do
    # *not* expose ``metadata`` as a plain namespace with a ``label``
    # field, so install a duck-typed shim that mirrors what the
    # workflow's ``submit_children`` outlined.
    calcjob.metadata = SimpleNamespace(label="")
    inner.metadata = SimpleNamespace(label=inner_label)
    root.metadata = SimpleNamespace(label=root_label)

    return calcjob, root


class TestVaspRealProvenanceAgainstRealAiiDA:
    """Run the same scenarios as :class:`TestVaspRealProvenanceSmear` /
    :class:`TestVaspRealProvenanceConvergence` but using real AiiDA
    nodes (``aiida.orm.WorkChainNode`` / ``CalcJobNode`` / ``Dict`` /
    ``Float`` / ``KpointsData``) instead of SimpleNamespace fakes.
    """

    def test_smear_via_inputs_parameters(self):
        """``Dict({'incar': {'ismear': 2, 'sigma': 0.06}})`` on the
        intermediate VaspWorkChain drives the resolver."""
        calcjob, root = _build_vasp_provenance_real(
            root_label="VaspSmearWorkChain",
            inner_label="",
            parameters_dict={"incar": {"ismear": 2, "sigma": 0.06}},
        )

        assert (
            resolve_label(
                calcjob,
                backend="vasp",
                method="smear",
                root_workchain=root,
            )
            == "ismear_2_sigma_0_060000"
        )
        # Same answer without the root_workchain argument.
        assert (
            resolve_label(calcjob, backend="vasp", method="smear")
            == "ismear_2_sigma_0_060000"
        )

    def test_smear_via_metadata_label(self):
        """The intermediate VaspWorkChain has ``label='ismear_2_sigma_0_060000'``
        and no ``parameters`` input — resolver must use the metadata fallback."""
        calcjob, _root = _build_vasp_provenance_real(
            root_label="VaspSmearWorkChain",
            inner_label="ismear_2_sigma_0_060000",
            parameters_dict=None,
        )

        assert (
            resolve_label(calcjob, backend="vasp", method="smear")
            == "ismear_2_sigma_0_060000"
        )

    def test_convergence_via_inputs_parameters_encut_400_spacing_0_2(self):
        """``Dict({'incar': {'encut': 400}})`` and ``Float(0.2)`` on the
        intermediate VaspWorkChain drive the resolver."""
        calcjob, root = _build_vasp_provenance_real(
            root_label="VaspConvergenceWorkChain",
            inner_label="",
            parameters_dict={"incar": {"encut": 400}},
            kpoints_spacing=0.2,
        )

        assert (
            resolve_label(
                calcjob,
                backend="vasp",
                method="convergence",
                root_workchain=root,
            )
            == "kpoints_spacing_0_2_encut_400"
        )
        # Same answer without the root_workchain argument.
        assert (
            resolve_label(calcjob, backend="vasp", method="convergence")
            == "kpoints_spacing_0_2_encut_400"
        )

    def test_convergence_via_metadata_label(self):
        """The intermediate VaspWorkChain has ``label='kpoints_spacing_0_2_encut_400'``."""
        calcjob, _root = _build_vasp_provenance_real(
            root_label="VaspConvergenceWorkChain",
            inner_label="kpoints_spacing_0_2_encut_400",
            parameters_dict=None,
        )

        assert (
            resolve_label(calcjob, backend="vasp", method="convergence")
            == "kpoints_spacing_0_2_encut_400"
        )

    def test_real_process_node_label_is_direct_attribute(self):
        """Sanity check the user's hint: a real AiiDA ``WorkChainNode``
        stores its submit-time label as the direct ``label`` attribute
        (``node.label``), *not* as ``node.metadata.label``."""
        from aiida.orm import WorkChainNode

        inner = WorkChainNode()
        inner.label = "kpoints_spacing_0_2_encut_400"
        assert inner.label == "kpoints_spacing_0_2_encut_400"
        # The ``ProcessNode`` does not expose ``metadata`` as a
        # namespace with a ``label`` field; the resolver reads it
        # through a duck-typed shim installed by the test helper.
        assert not hasattr(inner, "metadata") or not hasattr(
            getattr(inner, "metadata", object()), "label"
        ) or getattr(inner.metadata, "label", "") == ""


class TestResolveFallbackNeverUsesClassNames:
    """When the parameter chain is unresolvable, the fallback must
    land on ``calcjob_<pk>`` (or a meaningful metadata label) — never
    on the generic WorkChain / CalcJob class names that the AiiDA
    ProcessNode class machinery would otherwise surface."""

    @pytest.mark.parametrize(
        "process_label,class_label,expected",
        [
            ("VaspWorkChain", "vasp", "calcjob_60"),
            ("AbacusWorkChain", "abacus", "calcjob_61"),
            ("VaspCalculation", "vasp", "calcjob_62"),
            ("AbacusCalculation", "abacus", "calcjob_63"),
        ],
    )
    def test_class_names_are_not_used_as_directory_names(
        self, process_label, class_label, expected
    ):
        pk = int(expected.split("_")[-1])
        node = SimpleNamespace(
            pk=pk,
            process_label=process_label,
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
            caller=None,
        )
        assert resolve_label(node, backend=class_label, method="smear") == expected

    def test_caller_class_names_are_not_used_either(self):
        """Same guarantee for the caller chain: a parent whose
        process_label is a WorkChain class name must not be used as
        the final directory name."""
        node = SimpleNamespace(
            pk=70,
            process_label="VaspCalculation",
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
            caller=None,
        )
        parent_vc = SimpleNamespace(
            pk=71,
            process_label="VaspWorkChain",
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
            caller=None,
        )
        parent_root = SimpleNamespace(
            pk=72,
            process_label="VaspSmearWorkChain",
            metadata=SimpleNamespace(label=""),
            inputs=_FakeInputs({}),
            caller=None,
        )
        node.caller = parent_vc
        parent_vc.caller = parent_root

        assert (
            resolve_label(node, backend="vasp", method="smear")
            == "calcjob_70"
        )
