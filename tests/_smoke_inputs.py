"""Verify inner.inputs works with INPUT_WORK links."""
from aiida.orm import Dict, Float, KpointsData, WorkChainNode, CalcJobNode
from aiida.common import LinkType


def test_inner_inputs_works():
    inner = WorkChainNode()
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
    # Add caller chain
    calc = CalcJobNode()
    calc.base.links.add_incoming(inner, link_type=LinkType.CALL_CALC, link_label="call")
    print("inner.pk =", inner.pk)
    print("inner.inputs.parameters =", inner.inputs.parameters)
    print("inner.inputs.kpoints_spacing =", inner.inputs.kpoints_spacing)
    print("type(inner.inputs) =", type(inner.inputs))
    print("hasattr(inner.inputs, 'get') =", hasattr(inner.inputs, "get"))
    assert inner.inputs.parameters.get_dict() == {"incar": {"encut": 400}}
    assert inner.inputs.kpoints_spacing.value == 0.2


def test_inner_inputs_with_stored_node():
    """Stored nodes may behave differently."""
    inner = WorkChainNode()
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
    inner.store()
    print("inner.pk (stored) =", inner.pk)
    print("type(inner.inputs) (stored) =", type(inner.inputs))
    print("hasattr(inner.inputs, 'get') (stored) =", hasattr(inner.inputs, "get"))
    assert inner.inputs.parameters.get_dict() == {"incar": {"encut": 400}}
    assert inner.inputs.kpoints_spacing.value == 0.2