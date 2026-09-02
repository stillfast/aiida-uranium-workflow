"""Verify the relax_settings dicts the VASP elastic workflow sends
actually pass ``VaspRelaxWorkChain``'s spec validation (the real
runtime path — a rejected dict would except the relax child).

The workflow always sends the **complete** RelaxOptions defaults (with
any user overrides merged), because VaspRelaxWorkChain consumes the
dict verbatim as its runtime config.
"""

from __future__ import annotations

pytest_plugins = ["aiida.tools.pytest_fixtures"]

from aiida import orm
from aiida_vasp.utils.opthold import RelaxOptions
from aiida_vasp.workchains.v2.relax import VaspRelaxWorkChain


def _default_settings():
    return dict(RelaxOptions().model_dump())


def test_full_defaults_validate(aiida_profile):
    d = orm.Dict(dict=_default_settings())
    err = VaspRelaxWorkChain.spec().inputs["relax_settings"].validate(d)
    assert err is None


def test_defaults_merged_with_user_keys_validate(aiida_profile):
    settings = _default_settings()
    settings.update({"force_cutoff": 0.05, "steps": 40, "algo": "rd"})
    d = orm.Dict(dict=settings)
    err = VaspRelaxWorkChain.spec().inputs["relax_settings"].validate(d)
    assert err is None


def test_empty_dict_also_validates(aiida_profile):
    # (informational) — the spec accepts it, but the workchain's runtime
    # config would then miss e.g. ``perform``, which is why the workflow
    # never sends an empty dict.
    d = orm.Dict(dict={})
    err = VaspRelaxWorkChain.spec().inputs["relax_settings"].validate(d)
    assert err is None
