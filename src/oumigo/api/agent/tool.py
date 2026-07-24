"""Function-call *tools* — a Python callback plus the JSON Schema advertised to the model.

A tool is a plain Python function the agent loop may invoke on the model's behalf.
Rather than hand-write the OpenAI/vLLM function schema for each one, decorate the
function with ``@tool`` and the schema is **inferred** from its signature (parameter
types, which are required) and its docstring (the human descriptions):

    @tool
    def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
        '''Get the current weather for a city.

        Args:
            city: City name, e.g. Tokyo.
            units: Temperature unit.
        '''
        ...

For a tool whose parameters are too complex to infer, supply the schema yourself and
inference is skipped — but the override is still validated against the signature:

    @tool(parameters={"type": "object", "properties": {...}, "required": [...]})
    def search(query: str) -> str: ...

**Strict by default.** ``@tool`` validates the declaration at *decoration time* — i.e.
when the module is imported — and, if anything is wrong, raises a single
``ToolDefinitionError`` that lists *every* problem it found (with the source location),
so a malformed tool is caught the moment you load your code, never mid-conversation.
``@tool(strict=False)`` demotes only the *documentation* requirements (docstring and
per-parameter descriptions) to warnings; the structural checks — annotations, supported
types, no ``*args``/``**kwargs``, a valid override — remain hard errors regardless.

This module is deliberately self-contained: it has no dependency on a running manager
or worker, so the whole tool surface is unit-testable in isolation.
"""

from __future__ import annotations

import inspect
import logging
import re
import typing
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin

log = logging.getLogger("oumigo.api.agent.tool")

# OpenAI/vLLM constrain a function name to this shape.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Python scalar -> JSON Schema "type". bool must be checked before int (bool subtypes int).
_SCALAR_JSON: dict[type, str] = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
}

# JSON Schema "type" values an explicit `parameters=` override may use.
_JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}

# Return annotations we consider safe to serialize back to the model as a tool result.
_JSON_RETURN = {str, int, float, bool, list, dict, type(None)}

# First parameter names we skip when a bound-method-style function is decorated.
_SELF_NAMES = {"self", "cls"}


class ToolDefinitionError(TypeError):
    """A tool was declared incorrectly. Aggregates *all* problems found in one function.

    Raised at decoration time by ``@tool`` (and by :meth:`Tool.from_function`). The
    message names the offending function, its source location, and a bulleted list of
    every violation, so the whole declaration can be fixed in a single pass.
    """

    def __init__(self, fn: Callable[..., Any], problems: list[str]) -> None:
        self.fn = fn
        self.problems = problems
        where = _describe_location(fn)
        name = getattr(fn, "__name__", repr(fn))
        n = len(problems)
        header = f"{name} ({where}) has {n} problem{'s' if n != 1 else ''}:"
        body = "\n".join(f"  - {p}" for p in problems)
        hint = "\nFix them, or pass an explicit parameters={...} to bypass inference."
        super().__init__(f"{header}\n{body}{hint}")


@dataclass(frozen=True)
class Tool:
    """A callable exposed to the model as a function-call tool.

    Callable itself: ``tool(**kwargs)`` (or :meth:`invoke`) runs the underlying function,
    so a decorated function stays usable as a normal function while also carrying its
    schema. Build one with the :func:`tool` decorator or :meth:`from_function`; both run
    the same strict validation, so a ``Tool`` that exists is always well-formed.
    """

    fn: Callable[..., Any]
    name: str
    description: str
    parameters: dict[str, Any]

    def invoke(self, **kwargs: Any) -> Any:
        """Call the underlying function with the (already-parsed) keyword arguments."""
        return self.fn(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def to_openai(self) -> dict[str, Any]:
        """The OpenAI/vLLM ``tools=[...]`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        strict: bool = True,
    ) -> Tool:
        """Build (and strictly validate) a Tool from a Python function.

        The shared builder behind the :func:`tool` decorator. Infers the JSON Schema
        from the signature + docstring unless ``parameters`` is given, in which case the
        override is validated against the signature instead. Raises
        :class:`ToolDefinitionError` listing all problems if the declaration is invalid.
        """
        return _build_tool(
            fn, name=name, description=description, parameters=parameters, strict=strict
        )


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    strict: bool = True,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator turning a function into a :class:`Tool`, inferring its schema.

    Usable bare (``@tool``) or called (``@tool(strict=False)``, ``@tool(parameters=...)``).
    See the module docstring for the inference rules and the strict-validation contract.
    """
    def wrap(f: Callable[..., Any]) -> Tool:
        return _build_tool(
            f, name=name, description=description, parameters=parameters, strict=strict
        )

    return wrap if fn is None else wrap(fn)


# --------------------------------------------------------------------------- #
# Builder / validator
# --------------------------------------------------------------------------- #


def _build_tool(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    parameters: dict[str, Any] | None,
    strict: bool,
) -> Tool:
    """Validate a tool declaration, collecting every problem, then build the Tool.

    Documentation problems are collected separately: they are hard errors under the
    default ``strict=True`` and merely warnings under ``strict=False``. Structural
    problems are always hard errors.
    """
    if not callable(fn):
        raise ToolDefinitionError(fn, ["@tool must decorate a callable"])

    problems: list[str] = []       # structural — always fatal
    doc_problems: list[str] = []   # documentation — fatal only under strict

    resolved_name = name or getattr(fn, "__name__", "")
    if resolved_name == "<lambda>":
        problems.append("cannot infer a tool name from a lambda; use a named 'def' or pass name=")
    elif not _NAME_RE.match(resolved_name):
        problems.append(
            f"tool name {resolved_name!r} must match [A-Za-z0-9_-] and be 1-64 chars"
        )

    params = _signature_params(fn, problems)  # also flags *args/**kwargs/positional-only
    hints = _resolve_hints(fn)  # resolves string annotations (PEP 563 / `from __future__`)

    desc, doc_args = _parse_docstring(fn)
    resolved_desc = (description if description is not None else desc).strip()
    if not resolved_desc:
        doc_problems.append("missing a description (add a docstring, or pass description=)")

    if parameters is not None:
        _validate_override(parameters, params, problems)
    else:
        parameters = _infer_parameters(params, hints, doc_args, problems, doc_problems)

    _validate_return_annotation(fn, hints, problems)

    if strict:
        problems.extend(doc_problems)
    elif doc_problems:
        warnings.warn(
            f"{resolved_name}: " + "; ".join(doc_problems),
            stacklevel=3,
        )

    if problems:
        raise ToolDefinitionError(fn, problems)

    return Tool(fn=fn, name=resolved_name, description=resolved_desc, parameters=parameters)


def _signature_params(
    fn: Callable[..., Any], problems: list[str]
) -> list[inspect.Parameter]:
    """The tool-relevant parameters, flagging any that can't be represented/invoked.

    ``*args``/``**kwargs`` can't be a fixed JSON Schema and positional-only params can't
    be passed by keyword (the loop calls tools with a parsed JSON *object*), so all are
    fatal. A leading ``self``/``cls`` is dropped (bound-method-style functions)."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        problems.append("could not introspect the function signature")
        return []

    out: list[inspect.Parameter] = []
    for i, p in enumerate(sig.parameters.values()):
        if i == 0 and p.name in _SELF_NAMES and p.annotation is inspect.Parameter.empty:
            continue  # skip an unannotated leading self/cls
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            problems.append(f"'*{p.name}' is not representable as a tool parameter")
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            problems.append(f"'**{p.name}' is not representable as a tool parameter")
        elif p.kind is inspect.Parameter.POSITIONAL_ONLY:
            problems.append(
                f"parameter '{p.name}' is positional-only; tools are called by keyword"
            )
        else:
            out.append(p)
    return out


def _resolve_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Resolve annotations to real types, tolerating ones that can't be resolved.

    Under ``from __future__ import annotations`` (PEP 563) annotations reach us as
    strings; ``get_type_hints`` evaluates them. If evaluation fails (e.g. a name only
    imported inside the defining function), fall back to ``{}`` and let the per-parameter
    checks report the specific unsupported/missing types.
    """
    try:
        return typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolved annotations must not crash import
        return {}


def _infer_parameters(
    params: list[inspect.Parameter],
    hints: dict[str, Any],
    doc_args: dict[str, str],
    problems: list[str],
    doc_problems: list[str],
) -> dict[str, Any]:
    """Build the JSON Schema object from the signature + docstring, collecting problems."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for p in params:
        if p.name not in hints and p.annotation is inspect.Parameter.empty:
            problems.append(f"parameter '{p.name}': missing a type annotation")
            continue
        annotation = hints.get(p.name, p.annotation)
        schema, optional = _schema_for_annotation(annotation, p.name, problems)
        if schema is None:
            continue  # an error was already recorded

        doc = doc_args.get(p.name)
        if doc:
            schema = {**schema, "description": doc}
        else:
            doc_problems.append(f"parameter '{p.name}': undocumented (add it to the docstring Args)")

        has_default = p.default is not inspect.Parameter.empty
        if has_default:
            _check_default(p.name, annotation, p.default, problems)
        if not has_default and not optional:
            required.append(p.name)
        properties[p.name] = schema

    for documented in doc_args:
        if documented not in {p.name for p in params}:
            doc_problems.append(
                f"docstring documents '{documented}', which is not a parameter"
            )

    schema_obj: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema_obj["required"] = required
    return schema_obj


def _schema_for_annotation(
    annotation: Any, param_name: str, problems: list[str]
) -> tuple[dict[str, Any] | None, bool]:
    """Map a Python annotation to a JSON-Schema fragment.

    Returns ``(schema, optional)`` where ``optional`` is True for ``X | None`` /
    ``Optional[X]``. Returns ``(None, False)`` and records a problem for anything the
    strict inference doesn't support (``Any``, bare ``list``/``dict``, multi-type unions).
    """
    origin = get_origin(annotation)

    # Optional[X] / X | None -> the inner schema, marked optional.
    if origin is Union or _is_uniontype(annotation):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) != 1:
            problems.append(
                f"parameter '{param_name}': unions of multiple types aren't inferable; "
                f"pass an explicit parameters="
            )
            return None, False
        inner, _ = _schema_for_annotation(args[0], param_name, problems)
        return inner, True

    if origin is Literal:
        return _literal_schema(annotation, param_name, problems), False

    # Typed list[T].
    if origin in (list, typing.List):  # noqa: UP006 - handle both spellings
        item_args = get_args(annotation)
        if not item_args:
            problems.append(f"parameter '{param_name}': use list[T] with an item type, not bare list")
            return None, False
        item_schema, _ = _schema_for_annotation(item_args[0], param_name, problems)
        if item_schema is None:
            return None, False
        return {"type": "array", "items": item_schema}, False

    if annotation in _SCALAR_JSON:
        return {"type": _SCALAR_JSON[annotation]}, False

    if annotation in (list, dict) or origin is dict:
        problems.append(
            f"parameter '{param_name}': {_type_name(annotation)} can't be inferred; "
            f"pass an explicit parameters="
        )
        return None, False

    problems.append(
        f"parameter '{param_name}': unsupported type {_type_name(annotation)} "
        f"(supported: str, int, float, bool, list[T], Literal[...], Optional[...])"
    )
    return None, False


def _literal_schema(
    annotation: Any, param_name: str, problems: list[str]
) -> dict[str, Any] | None:
    """A ``Literal[...]`` becomes an enum; its members must share one scalar JSON type."""
    values = list(get_args(annotation))
    json_types = {_SCALAR_JSON.get(type(v)) for v in values}
    if None in json_types or len(json_types) != 1:
        problems.append(
            f"parameter '{param_name}': Literal members must share one scalar type "
            f"(all str, or all int, ...)"
        )
        return None
    return {"type": json_types.pop(), "enum": values}


def _check_default(param_name: str, annotation: Any, default: Any, problems: list[str]) -> None:
    """A scalar default must match its declared type (bool checked before int)."""
    if default is None:
        return  # a None default just means optional; covered elsewhere
    if annotation in _SCALAR_JSON:
        if annotation is int and isinstance(default, bool):
            problems.append(f"parameter '{param_name}': default {default!r} is bool, not int")
        elif not isinstance(default, annotation):
            problems.append(
                f"parameter '{param_name}': default {default!r} doesn't match declared "
                f"type {_type_name(annotation)}"
            )


def _validate_return_annotation(
    fn: Callable[..., Any], hints: dict[str, Any], problems: list[str]
) -> None:
    """The return is fed back to the model, so require an annotation of a serializable type."""
    try:
        raw = inspect.signature(fn).return_annotation
    except (ValueError, TypeError):
        return  # signature already flagged upstream
    if raw is inspect.Signature.empty:
        problems.append("missing a return type annotation (e.g. '-> str')")
        return
    ret = hints.get("return", raw)  # resolve the string form under PEP 563
    base = get_origin(ret) or ret
    if base not in _JSON_RETURN:
        problems.append(
            f"return type {_type_name(ret)} isn't JSON-serializable for a tool result "
            f"(use str/int/float/bool/list/dict)"
        )


def _validate_override(
    parameters: dict[str, Any], params: list[inspect.Parameter], problems: list[str]
) -> None:
    """Validate a hand-supplied ``parameters=`` schema against the function signature.

    An explicit override skips *inference*, not *validation*: the schema must be a proper
    object schema, every property must carry a type and description, ``required`` must
    reference real properties, and the property names must match the signature exactly.
    """
    if not isinstance(parameters, dict):
        problems.append("parameters= override must be a dict (JSON Schema object)")
        return
    if parameters.get("type") != "object":
        problems.append("parameters= override must have type: 'object'")
    props = parameters.get("properties")
    if not isinstance(props, dict):
        problems.append("parameters= override must have a 'properties' dict")
        return

    for pname, pschema in props.items():
        if not isinstance(pschema, dict):
            problems.append(f"parameters= property '{pname}' must be a schema object")
            continue
        if pschema.get("type") not in _JSON_TYPES:
            problems.append(f"parameters= property '{pname}' needs a valid JSON 'type'")
        if not str(pschema.get("description", "")).strip():
            problems.append(f"parameters= property '{pname}' needs a non-empty 'description'")

    required = parameters.get("required", [])
    if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
        problems.append("parameters= 'required' must be a list of strings")
    else:
        for r in required:
            if r not in props:
                problems.append(f"parameters= 'required' names '{r}', absent from properties")

    sig_names = {p.name for p in params}
    prop_names = set(props)
    if missing := sig_names - prop_names:
        problems.append(
            f"parameters= is missing propert{'ies' if len(missing) > 1 else 'y'} for "
            f"{', '.join(sorted(missing))}"
        )
    if extra := prop_names - sig_names:
        problems.append(
            f"parameters= describes unknown propert{'ies' if len(extra) > 1 else 'y'} "
            f"{', '.join(sorted(extra))}"
        )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _is_uniontype(annotation: Any) -> bool:
    """True for the ``X | Y`` (PEP 604) form, whose origin isn't ``typing.Union``."""
    return type(annotation).__name__ == "UnionType"


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))


def _describe_location(fn: Callable[..., Any]) -> str:
    """``file.py:LINE`` for the function, or a best-effort fallback."""
    try:
        code = fn.__code__  # type: ignore[attr-defined]
        return f"{code.co_filename}:{code.co_firstlineno}"
    except AttributeError:
        module = getattr(fn, "__module__", "?")
        return module


_SECTION_RE = re.compile(r"^(Args|Arguments|Parameters|Returns|Raises|Yields|Examples?):\s*$")
_ARG_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring(fn: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (description, {param: description}).

    The description is everything before the first section header; the arg map comes from
    the ``Args:`` section (continuation lines are folded into the preceding entry).
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return "", {}

    lines = doc.splitlines()
    desc_lines: list[str] = []
    args: dict[str, str] = {}
    i = 0
    n = len(lines)

    # Description: up to the first recognized section header.
    while i < n and not _SECTION_RE.match(lines[i].strip()):
        desc_lines.append(lines[i])
        i += 1

    # Walk sections; only Args-like ones are parsed.
    while i < n:
        header = _SECTION_RE.match(lines[i].strip())
        i += 1
        if not header or header.group(1) not in ("Args", "Arguments", "Parameters"):
            continue
        current: str | None = None
        while i < n and not _SECTION_RE.match(lines[i].strip()):
            raw = lines[i]
            i += 1
            stripped = raw.strip()
            if not stripped:
                continue
            m = _ARG_RE.match(stripped)
            if m:
                current = m.group(1)
                args[current] = m.group(2).strip()
            elif current is not None:  # continuation of the previous arg's description
                args[current] = (args[current] + " " + stripped).strip()

    return "\n".join(desc_lines).strip(), args
