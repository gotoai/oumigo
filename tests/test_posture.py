"""Providers (LAN / CUSTOM) and the CUSTOM network-posture rules."""

from __future__ import annotations

import pytest

from oumigo.providers import KNOWN_PROVIDERS, create_provider
from oumigo.service.manager.posture import check_posture, is_loopback


def test_custom_provider_is_known_and_case_insensitive() -> None:
    assert KNOWN_PROVIDERS == ("LAN", "CUSTOM")
    custom = create_provider("custom")
    assert custom.name == "CUSTOM" and custom.is_wan
    lan = create_provider("LAN")
    assert lan.name == "LAN" and not lan.is_wan
    with pytest.raises(ValueError):
        create_provider("runpod")


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("127.0.1.5", True), ("::1", True), ("localhost", True),
    ("0.0.0.0", False), ("192.168.1.10", False), ("oumigo.example.net", False),
])
def test_is_loopback(host: str, expected: bool) -> None:
    assert is_loopback(host) is expected


def _check(provider: str, **overrides) -> list[str]:
    layout = dict(control_host="127.0.0.1", control_tls=False, data_host="127.0.0.1",
                  client_auth=True)
    layout.update(overrides)
    return check_posture(provider, **layout)


def test_lan_allows_any_bind() -> None:
    assert _check("LAN", control_host="0.0.0.0", data_host="0.0.0.0", client_auth=False) == []


def test_custom_loopback_layout_is_ok() -> None:
    assert _check("CUSTOM") == []


def test_custom_public_control_plane_needs_tls() -> None:
    assert len(_check("CUSTOM", control_host="0.0.0.0")) == 1
    assert _check("CUSTOM", control_host="0.0.0.0", control_tls=True) == []


def test_custom_router_must_stay_local() -> None:
    problems = _check("CUSTOM", data_host="0.0.0.0")
    assert len(problems) == 1 and "data_plane.host" in problems[0]


@pytest.mark.parametrize("provider", ["LAN", "CUSTOM"])
def test_web_console_binds_loopback_only_on_any_provider(provider: str) -> None:
    from oumigo.service.manager.settings import get_dashboard

    assert get_dashboard({"provider": provider}) == (True, "127.0.0.1", 7080)  # default
    assert get_dashboard({"dashboard": {"host": "::1", "port": 7081}}) == (True, "::1", 7081)
    with pytest.raises(ValueError, match="binds loopback only"):
        get_dashboard({"dashboard": {"host": "0.0.0.0"}})
    # Disabled, its bind address doesn't matter.
    assert get_dashboard({"dashboard": {"enabled": False, "host": "0.0.0.0"}})[0] is False


def test_custom_router_requires_client_auth() -> None:
    problems = _check("CUSTOM", client_auth=False)
    assert len(problems) == 1 and "client_auth" in problems[0]


@pytest.mark.parametrize("config,expected", [
    ({"provider": "CUSTOM"}, True),
    ({"provider": "custom"}, True),
    ({}, False),                                                  # LAN by default
    ({"provider": "LAN", "data_plane": {"client_auth": True}}, True),
    ({"provider": "CUSTOM", "data_plane": {"client_auth": False}}, False),  # posture refuses it
])
def test_client_auth_defaults_by_provider(config: dict, expected: bool) -> None:
    from oumigo.service.manager.settings import get_client_auth

    assert get_client_auth(config) is expected
