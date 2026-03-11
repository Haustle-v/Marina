from typing import Any, Dict, List, Optional


class _FakeRay:
    def __init__(self) -> None:
        self._inited = False
        self.init_calls: List[Dict[str, Any]] = []

    def is_initialized(self) -> bool:
        return self._inited

    def init(self, *args, **kwargs) -> None:
        self.init_calls.append({"args": args, "kwargs": kwargs})
        self._inited = True


def test_ensure_ray_initialized_local_defaults():
    from pyseekdb.client.db import _ensure_ray_initialized

    ray = _FakeRay()
    _ensure_ray_initialized(ray, ray_address=None, ray_init_kwargs=None)

    assert ray.is_initialized()
    assert len(ray.init_calls) == 1
    kw = ray.init_calls[0]["kwargs"]
    assert kw["ignore_reinit_error"] is True
    assert kw["include_dashboard"] is False


def test_ensure_ray_initialized_remote_address():
    from pyseekdb.client.db import _ensure_ray_initialized

    ray = _FakeRay()
    _ensure_ray_initialized(
        ray,
        ray_address="ray://127.0.0.1:10001",
        ray_init_kwargs={"namespace": "pyseekdb"},
    )

    assert ray.is_initialized()
    assert len(ray.init_calls) == 1
    kw = ray.init_calls[0]["kwargs"]
    assert kw["address"] == "ray://127.0.0.1:10001"
    assert kw["namespace"] == "pyseekdb"
    assert kw["ignore_reinit_error"] is True




