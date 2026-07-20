"""The manager owns a stable Worker#N <-> UUID mapping, assigned at registration."""

from __future__ import annotations

from oumigo.manager.control.registry import Registry


def _register(reg: Registry, node_id: str):
    return reg.register(
        node_id=node_id,
        address="10.0.0.1:7001",
        state="REGISTERED",
        incarnation=1,
        capabilities={},
    )


def test_sequential_names_in_registration_order() -> None:
    reg = Registry()
    a = _register(reg, "uuid-a")
    b = _register(reg, "uuid-b")
    assert a.name == "Worker#1"
    assert b.name == "Worker#2"


def test_reregister_keeps_the_same_name() -> None:
    reg = Registry()
    _register(reg, "uuid-a")
    _register(reg, "uuid-b")
    again = _register(reg, "uuid-a")  # re-register the first node
    assert again.name == "Worker#1"
    # a genuinely new node still gets the next number, not a reused one
    assert _register(reg, "uuid-c").name == "Worker#3"


def test_name_is_exposed_in_as_dict() -> None:
    reg = Registry()
    rec = _register(reg, "uuid-a")
    d = rec.as_dict()
    assert d["seq"] == 1
    assert d["name"] == "Worker#1"
