"""Experimental HCU gates must agree with the compiler cache identity."""

import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton import knobs
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.runtime import cache

FEATURES = [
    ("FLAGTREE_HCU_EXPERIMENTAL_SOFTWARE_WS", "experimental_software_ws", "gpu.warp_specialize"),
    ("FLAGTREE_HCU_EXPERIMENTAL_STATIC_PIPE", "experimental_static_pipe", "pipe"),
]


@pytest.fixture(autouse=True)
def clear_feature_flags(monkeypatch):
    for variable, _, _ in FEATURES:
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize("variable,knob,primitive", FEATURES)
@pytest.mark.parametrize("value,enabled,normalized", [
    (None, False, None),
    ("0", False, "false"),
    ("false", False, "false"),
    ("OFF", False, "false"),
    ("1", True, "true"),
    ("true", True, "true"),
    ("True", True, "true"),
    ("on", True, "true"),
    ("ON", True, "true"),
])
def test_feature_gate_matches_cache_normalization(variable, knob, primitive, value, enabled, normalized, monkeypatch):
    if value is not None:
        monkeypatch.setenv(variable, value)

    assert getattr(knobs.hcu, knob) is enabled
    assert get_cache_invalidating_env_vars().get(variable) == normalized

    # Run the capability declaration afresh without mutating its imported
    # module. Feature flags are configured before loading the backend.
    declaration = Path(__file__).resolve().parents[3] / "backend" / "tle_supported.py"
    supported = runpy.run_path(str(declaration))["TLE_SUPPORTED_PRIMITIVES"]
    assert (primitive in supported) is enabled


@pytest.mark.parametrize("variable,knob,primitive", FEATURES)
@pytest.mark.parametrize("disabled", [None, "0", "false", "OFF"])
def test_opt_out_does_not_reuse_opt_in_cache(variable, knob, primitive, disabled, monkeypatch, fresh_triton_cache):
    # Hold the source, compiler and options fixed, as in two compiler processes
    # sharing one disk cache. No HCU compiler or device is needed for this check.
    component = SimpleNamespace(hash=lambda: "unchanged")
    monkeypatch.setattr(cache, "triton_key", lambda: "same-build")

    def cache_for_environment():
        key = cache.get_cache_key(component, component, component, get_cache_invalidating_env_vars())
        return cache.FileCacheManager(hashlib.sha256(key.encode()).hexdigest())

    monkeypatch.setenv(variable, "1")
    enabled_cache = cache_for_environment()
    enabled_path = enabled_cache.put("enabled artifact", "kernel.json", binary=False)
    enabled_cache.put_group("kernel.json", {"kernel.json": enabled_path})
    assert enabled_cache.get_group("kernel.json") is not None

    if disabled is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, disabled)

    disabled_cache = cache_for_environment()
    assert disabled_cache.key != enabled_cache.key
    assert disabled_cache.get_group("kernel.json") is None

    monkeypatch.setenv(variable, "1")
    assert cache_for_environment().get_group("kernel.json") == {"kernel.json": enabled_path}
