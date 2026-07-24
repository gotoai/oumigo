"""Unit tests for the tool-definition surface (oumigo.api.agent.tool).

These exercise the strict ``@tool`` validator in isolation — no manager or worker is
involved. They cover schema inference from the signature + docstring, the explicit
``parameters=`` override path, and that a malformed declaration fails fast at decoration
time with an aggregated ``ToolDefinitionError``.
"""

from __future__ import annotations

from typing import Literal, Optional

import pytest

from oumigo.api.agent.tool import Tool, ToolDefinitionError, tool


# --------------------------------------------------------------------------- #
# Inference — the happy path
# --------------------------------------------------------------------------- #


def test_infers_schema_from_signature_and_docstring():
    """Types come from the signature; descriptions from the docstring Args section."""

    @tool
    def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
        """Get the current weather for a city.

        Args:
            city: City name, e.g. Tokyo.
            units: Temperature unit.
        """
        return f"sunny in {city} ({units})"

    assert isinstance(get_weather, Tool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the current weather for a city."

    props = get_weather.parameters["properties"]
    assert props["city"] == {"type": "string", "description": "City name, e.g. Tokyo."}
    assert props["units"]["type"] == "string"
    assert props["units"]["enum"] == ["celsius", "fahrenheit"]
    # city has no default -> required; units has one -> optional.
    assert get_weather.parameters["required"] == ["city"]


def test_tool_stays_callable_and_exposes_openai_schema():
    """A decorated function remains callable and can emit the OpenAI tools[] entry."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: First addend.
            b: Second addend.
        """
        return a + b

    assert add(2, 3) == 5           # __call__ delegates to the function
    assert add.invoke(a=2, b=3) == 5

    entry = add.to_openai()
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "add"
    assert set(entry["function"]["parameters"]["required"]) == {"a", "b"}
    assert entry["function"]["parameters"]["properties"]["a"]["type"] == "integer"


def test_type_mapping_and_optional_and_list():
    """Scalars map to JSON types; Optional[X] is not required; list[T] is an array."""

    @tool
    def f(name: str, count: int, ratio: float, flag: bool,
          tags: list[str], nickname: Optional[str] = None) -> dict:
        """Exercise the supported type mappings.

        Args:
            name: A string.
            count: An integer.
            ratio: A float.
            flag: A boolean.
            tags: A list of strings.
            nickname: An optional string.
        """
        return {}

    props = f.parameters["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert props["tags"] == {"type": "array", "items": {"type": "string"},
                             "description": "A list of strings."}
    # Optional[str] with a None default -> present but not in `required`.
    assert props["nickname"]["type"] == "string"
    assert "nickname" not in f.parameters["required"]
    assert set(f.parameters["required"]) == {"name", "count", "ratio", "flag", "tags"}


def test_multiline_arg_description_is_folded():
    """A wrapped Args description spanning several lines is joined into one string."""

    @tool
    def q(query: str) -> str:
        """Search.

        Args:
            query: The search query string that
                can wrap across lines.
        """
        return ""

    assert q.parameters["properties"]["query"]["description"] == (
        "The search query string that can wrap across lines."
    )


# --------------------------------------------------------------------------- #
# Strictness — aggregated, decoration-time failures
# --------------------------------------------------------------------------- #


def test_missing_annotation_is_a_hard_error():
    with pytest.raises(ToolDefinitionError, match="missing a type annotation"):

        @tool
        def f(city) -> str:  # noqa: ANN001 - intentionally missing
            """Do a thing.

            Args:
                city: A city.
            """
            return ""


def test_all_problems_are_reported_together():
    """One ToolDefinitionError lists every violation, not just the first."""
    with pytest.raises(ToolDefinitionError) as exc:

        @tool
        def f(city, units: str = "c"):  # missing annotation, no return, undocumented units
            """A tool."""
            return ""

    problems = exc.value.problems
    assert any("missing a type annotation" in p for p in problems)      # city
    assert any("return type annotation" in p for p in problems)          # no -> ...
    assert any("undocumented" in p for p in problems)                    # units
    assert len(problems) >= 3


def test_unsupported_type_is_rejected():
    import datetime

    with pytest.raises(ToolDefinitionError, match="unsupported type"):

        @tool
        def f(when: datetime.datetime) -> str:
            """Do a thing.

            Args:
                when: A time.
            """
            return ""


def test_var_args_and_kwargs_are_rejected():
    with pytest.raises(ToolDefinitionError, match="not representable"):

        @tool
        def f(*args: int) -> str:
            """Bad.

            Args:
                args: nope.
            """
            return ""


def test_lambda_is_rejected():
    with pytest.raises(ToolDefinitionError, match="lambda"):
        tool(lambda x: x)


def test_default_type_mismatch_is_rejected():
    with pytest.raises(ToolDefinitionError, match="doesn't match declared"):

        @tool
        def f(n: int = "three") -> str:  # noqa: A002
            """Bad default.

            Args:
                n: A number.
            """
            return ""


def test_docstring_documents_unknown_parameter():
    with pytest.raises(ToolDefinitionError, match="not a parameter"):

        @tool
        def f(city: str) -> str:
            """Drift.

            Args:
                city: A city.
                country: A country that isn't a parameter.
            """
            return ""


def test_missing_return_annotation_is_rejected():
    with pytest.raises(ToolDefinitionError, match="return type annotation"):

        @tool
        def f(city: str):
            """No return type.

            Args:
                city: A city.
            """
            return ""


def test_missing_docstring_is_error_under_strict_but_warning_when_relaxed():
    # Strict (default): a hard error.
    with pytest.raises(ToolDefinitionError, match="description"):

        @tool
        def f(city: str) -> str:
            return ""

    # Relaxed: builds, but warns about the missing docs.
    with pytest.warns(UserWarning):

        @tool(strict=False)
        def g(city: str) -> str:
            return ""

    assert isinstance(g, Tool)
    # Structural checks still bite even when relaxed.
    with pytest.raises(ToolDefinitionError, match="missing a type annotation"):

        @tool(strict=False)
        def h(city) -> str:  # noqa: ANN001
            return ""


# --------------------------------------------------------------------------- #
# Explicit parameters= override
# --------------------------------------------------------------------------- #


def test_explicit_override_bypasses_inference():
    """An explicit schema is used verbatim (still validated against the signature)."""
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The query."}},
        "required": ["query"],
    }

    @tool(parameters=schema, description="Search the web.")
    def search(query: str) -> str:
        return ""

    assert search.parameters is schema
    assert search.description == "Search the web."


def test_override_property_names_must_match_signature():
    with pytest.raises(ToolDefinitionError, match="unknown propert"):

        @tool(
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string", "description": "d"}},
            },
            description="d",
        )
        def search(query: str) -> str:  # signature says 'query', schema says 'q'
            return ""


def test_override_property_requires_description():
    with pytest.raises(ToolDefinitionError, match="non-empty 'description'"):

        @tool(
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},  # no description
            },
            description="d",
        )
        def search(query: str) -> str:
            return ""


def test_from_function_matches_decorator():
    """Tool.from_function is the same builder the decorator uses."""

    def add(a: int, b: int) -> int:
        """Add.

        Args:
            a: x.
            b: y.
        """
        return a + b

    built = Tool.from_function(add)
    assert isinstance(built, Tool)
    assert built.name == "add"
    assert set(built.parameters["required"]) == {"a", "b"}
