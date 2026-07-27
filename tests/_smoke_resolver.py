"""Trace resolve_label with real AiiDA nodes."""
from aiida.orm import Dict, Float, KpointsData, WorkChainNode, CalcJobNode
from aiida.common import LinkType
from types import SimpleNamespace


def test_trace_resolver_convergence():
    calcjob = CalcJobNode()
    inner = WorkChainNode()
    root = WorkChainNode()

    inner.base.links.add_incoming(
        Dict({"incar": {"encut": 400}}),
        link_type=LinkType.INPUT_WORK,
        link_label="parameters",
    )
    inner.base.links.add_incoming(
        Float(0.2),
        link_type=LinkType.INPUT_WORK,
        link_label="kpoints_spacing",
    )

    calcjob.base.links.add_incoming(inner, link_type=LinkType.CALL_CALC, link_label="call")
    inner.base.links.add_incoming(root, link_type=LinkType.CALL_WORK, link_label="call")

    root.label = "VaspConvergenceWorkChain"
    inner.label = ""
    calcjob.label = ""

    calcjob.metadata = SimpleNamespace(label="")
    inner.metadata = SimpleNamespace(label="")
    root.metadata = SimpleNamespace(label="VaspConvergenceWorkChain")

    print("calc.caller =", calcjob.caller)
    print("inner.called =", inner.called)
    print("root.called =", root.called)
    print("inner.pk =", inner.pk)
    print("calcjob.pk =", calcjob.pk)
    print("root.pk =", root.pk)

    # Inspect inner.inputs directly
    print("inner.inputs.parameters =", inner.inputs.parameters)
    print("inner.inputs.kpoints_spacing =", inner.inputs.kpoints_spacing)
    print("hasattr(inner.inputs, 'get') =", hasattr(inner.inputs, "get"))

    from aiida_uranium_workflow.utils.labels import (
        _get_inputs_safely,
        _extract_incar,
        _read_vasp_kpoints,
        _candidate_nodes,
        _resolve_vasp_convergence,
        _walk_caller_ancestors,
    )
    inputs = _get_inputs_safely(inner)
    print("inner inputs (resolver view) =", inputs)
    print("inner inputs.parameters via getattr =", getattr(inputs, "parameters", None))
    print("hasattr(inputs, 'get') =", hasattr(inputs, "get"))

    candidates = _candidate_nodes(calcjob, root)
    print("candidates =", [type(n).__name__ + ":" + str(n.pk) for n in candidates])

    # Test extract_incar
    print("_extract_incar(inputs) =", _extract_incar(inputs))

    # Test kpoints
    kp, mode = _read_vasp_kpoints(inner, None)
    print("_read_vasp_kpoints(inner, None) =", kp, mode)