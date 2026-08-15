"""Dystopic entrypoint for the arklex Shopify customer-service assistant.

The platform calls ``run(task_input, *, proxy_url, run_token)`` and grades the
returned ``final_response``.

WHAT IS UNDER TEST
------------------
The repo's own agent stack: the ``openai-agent`` node (OpenAI Agents SDK
``Runner``), the orchestrator/task-graph, `Tool.execute` slot filling and
argument merging, and the customer-service prompt from
``examples/shopify/taskgraph.json``. Only the *outermost* I/O of each Shopify
tool is redirected to the simulated world.

DELIBERATE DEVIATIONS FROM THE CUSTOMER'S CODE
----------------------------------------------
Each of these is also recorded in ARKLEX_EGRESS_AUDIT.md.

1. **Tools are proxied at the registered-function boundary.** Every
   ``shopify/*`` tool's ``Tool.func`` is replaced with a shim that calls the
   Odyssey proxy and renders the structured response into the same
   ``message_flow`` / ``response`` output model the real tool returns. The
   consequence: the customer's Shopify **GraphQL envelope parsing** -
   ``json.loads(response)["data"]``, the ``userErrors`` branch, and the
   ``ToolExecutionError``/``ShopifyError`` raises, does **not** run. That code
   is therefore NOT under test.

   Why not intercept lower, at ``shopify.GraphQL().execute``? Because that
   method returns a JSON *string*. A tool's ``output_schema`` can pin the shape
   of a structured response but cannot pin the shape of text inside a string,
   so intercepting there would give up schema validation on every tool, the
   single most load-bearing defence against a simulator inventing a payload the
   agent cannot parse.

2. **``shopify`` (ShopifyAPI) is installed but never called.** It is imported
   at module scope by every tool, and ``resource_map`` silently *drops* any
   tool whose module fails to import, so the package must be present for the
   nine tools to exist at all.

3. **``get_order_details`` ignores ``order_ids`` / ``order_names``.** The
   declared ledger projection filters on ``user_id`` only and returns the
   customer's orders newest-first; the real tool additionally narrows by id or
   name. The agent still sees every order it needs, one of which it selects.

4. **The task graph is rebuilt.** ``dystopic/taskgraph.json`` replaces
   ``examples/shopify/taskgraph.json``, which does not load against HEAD (see
   ``dystopic/build_taskgraph.py`` for the full explanation). Role, objective,
   intro and the nine tools are carried over verbatim.

5. **``run.py`` is bypassed.** The repo's CLI entrypoint calls
   ``Executor(agents=...)`` and ``AgentOrg(config=..., env=...)``; neither
   keyword exists on the current signatures, so ``run.py`` raises ``TypeError``
   before reaching the orchestrator. This entrypoint calls the working
   signatures directly (``Executor(tools, workers, nodes, llm_config)`` and
   ``AgentOrg(config, executor)``).

HARNESS VARIANTS (CONFIGURATIONS)
---------------------------------
arklex is an agent *builder*: a config compiles into a task graph, so one repo
at one commit yields many different agents. The platform models that with a
**harness variant**, and hands the frozen configuration to this entrypoint as
``task_input["harness_variant"]``::

    {"snapshot_id": 12, "variant_id": 3, "name": "read-only-concierge",
     "fingerprint": "…", "knob_values": {"mutations_enabled": false, …}}

``dystopic_harness.py`` is the adapter: it validates those knob values and
compiles them into a taskgraph, so the tool surface, the agent prompt and the
model all follow the configuration. Two rules matter:

* **The key is absent when no variant was frozen**, and this entrypoint then
  loads the static ``dystopic/taskgraph.json`` exactly as it did before the
  feature existed. That path is unchanged on purpose.
* **An unbuildable configuration raises**, it does not fall back to defaults.
  Config-loading is part of what the check exercises; silently running the
  default agent under a variant's name would report a green suite for a
  configuration that never actually ran.

``configuration`` is read as a DEPRECATED ALIAS. The platform originally sent
the block under that name and renamed it to ``harness_variant`` mid-flight,
which silently unhooked this seam: the key this file read simply stopped
arriving, and since an absent key legitimately means "no variant was frozen"
the run fell through to the static taskgraph and registered all nine tools —
the exact green-suite-for-a-configuration-that-never-ran failure the rule
above exists to prevent. Absence cannot distinguish "no variant" from "wrong
key", so reading both names is the only defence this side of the wire. The
platform now also verifies the acknowledgement in ``metadata.harness_variant``
against the snapshot it froze, which is what makes a future desync loud.

WHY A MODULE-LEVEL ENVELOPE
---------------------------
``Tool.execute`` dispatches sync tool functions with
``asyncio.to_thread(self.func, ...)``. ``proxy_call`` resolves its envelope
from a ContextVar, and ContextVars are **not** inherited by worker threads, so
``proxy_call`` fails there. ``proxy_call_with`` + a module-level global is the
supported shape for thread-dispatched tools.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOT `dystopic.harness`: the SDK installs a real package named `dystopic`, which
# wins over this repo's `dystopic/` namespace directory, so a module placed there
# is unimportable in the sandbox. Top-level module, mirroring `dystopic_entry`.
from dystopic_harness import (  # noqa: E402
    ConfigurationError,
    build_taskgraph,
    load_configuration,
    tool_surface,
)
from dystopic.odyssey import Envelope, proxy_call_with  # noqa: E402

# Globals, not ContextVars, see module docstring.
_ENVELOPE: Envelope | None = None
_CALLS: list[dict[str, Any]] = []

TASKGRAPH = REPO_ROOT / "dystopic" / "taskgraph.json"


# --------------------------------------------------------------------------
# Response rendering: structured proxy response -> the customer's output model
#
# Each renderer mirrors the formatting the real tool produces, so the agent
# sees text in the shape its prompt and downstream code already expect.
# --------------------------------------------------------------------------

def _fmt_money(amount: Any, currency: Any) -> str:
    if amount is None:
        return "None"
    return f"{amount} {currency or ''}".strip()


def _render_user_details(resp: dict) -> str:
    # Real tool: message_flow=json.dumps(data)
    return json.dumps(resp)


def _render_order_details(resp: dict) -> str:
    orders = resp.get("orders") or []
    if not orders:
        return "You have no orders placed."
    out = ""
    for o in orders:
        out += f"Order ID: {o.get('order_id', 'None')}\n"
        out += f"Order Name: {o.get('order_name', 'None')}\n"
        out += f"Created At: {o.get('created_at', 'None')}\n"
        out += f"Cancelled At: {o.get('cancelled_at', 'None')}\n"
        out += f"Order Status: {o.get('status', 'None')}\n"
        out += f"Fulfillment Status: {o.get('fulfillment_status', 'None')}\n"
        out += f"Return Status: {o.get('return_status', 'None')}\n"
        out += f"Cancellable: {o.get('cancellable', 'None')}\n"
        if o.get("cancel_block_reason"):
            out += f"Cancel Block Reason: {o.get('cancel_block_reason')}\n"
        out += f"Financial Status: {o.get('financial_status', 'None')}\n"
        out += f"Total Price: {_fmt_money(o.get('total_amount'), o.get('currency'))}\n"
        out += "Line Items:\n"
        for item in o.get("line_items") or []:
            if not isinstance(item, dict):
                out += f"    {item}\n"
                continue
            out += f"    Title: {item.get('title', 'None')}\n"
            out += f"    Variant ID: {item.get('variant_id', 'None')}\n"
            out += f"    Quantity: {item.get('quantity', 'None')}\n"
            out += f"    Unit Price: {item.get('unit_price', 'None')}\n"
        out += "\n"
    return out


def _render_products(resp: dict) -> str:
    out = ""
    for p in resp.get("products") or []:
        out += f"Product ID: {p.get('product_id', 'None')}\n"
        out += f"Title: {p.get('title', 'None')}\n"
        out += f"Description: {p.get('description', 'None')}\n"
        out += f"Price: {_fmt_money(p.get('price'), p.get('currency'))}\n"
        out += f"Inventory Quantity: {p.get('inventory_quantity', 'None')}\n"
        out += f"Final Sale: {p.get('final_sale', 'None')}\n"
        out += f"Variant IDs: {p.get('variant_ids', 'None')}\n"
        out += "\n"
    return out or "No products found."


def _render_web_product(resp: dict) -> str:
    return _render_products({"products": [resp]})


def _render_search_products(resp: dict) -> str:
    # Real tool: response=json.dumps({"answer": ..., "card_list": ...})
    products = resp.get("products") or []
    card_list = [
        {
            "id": p.get("product_id"),
            "title": p.get("title"),
            "price": p.get("price"),
            "variant_ids": p.get("variant_ids"),
        }
        for p in products
    ]
    answer = (
        f"Found {len(products)} matching product(s)."
        if products
        else "No products matched that search."
    )
    return json.dumps({"answer": answer, "card_list": card_list})


def _render_cart(resp: dict) -> str:
    out = f"Cart ID: {resp.get('cart_id', 'None')}\n"
    out += f"Subtotal: {_fmt_money(resp.get('subtotal'), resp.get('currency'))}\n"
    for item_id in resp.get("item_ids") or []:
        out += f"Product Variant ID: {item_id}\n"
    return out


def _render_cart_add(resp: dict) -> str:
    return "Items are successfully added to the shopping cart. " + json.dumps(resp)


def _render_cancel_order(resp: dict) -> str:
    # Real tool: "The order is successfully cancelled. " + json.dumps(response)
    if resp.get("cancelled"):
        return "The order is successfully cancelled. " + json.dumps(resp)
    return "The order was not cancelled. " + json.dumps(resp)


def _render_return_products(resp: dict) -> str:
    if resp.get("accepted"):
        return "The product return request is successfully submitted. " + json.dumps(resp)
    return "The product return request was not accepted. " + json.dumps(resp)


# slug -> (proxy tool name, renderer, output field)
TOOL_SPECS: dict[str, tuple[str, Any, str]] = {
    "shopify/get-user-details-admin": ("get_user_details_admin", _render_user_details, "message_flow"),
    "shopify/get-order-details": ("get_order_details", _render_order_details, "message_flow"),
    "shopify/get-products": ("get_products", _render_products, "message_flow"),
    "shopify/get-web-product": ("get_web_product", _render_web_product, "message_flow"),
    "shopify/search-products": ("search_products", _render_search_products, "response"),
    "shopify/get-cart": ("get_cart", _render_cart, "message_flow"),
    "shopify/cart-add-items": ("cart_add_items", _render_cart_add, "message_flow"),
    "shopify/return-products": ("return_products", _render_return_products, "message_flow"),
    "shopify/cancel-order": ("cancel_order", _render_cancel_order, "message_flow"),
}

# Kwargs the framework injects that are not tool arguments.
_NON_TOOL_KWARGS = {
    "auth",
    "node_specific_data",
    "slots",
    "llm_provider",
    "model_type_or_path",
    "langchain_model_kwargs",
    "openai_agent_sdk_model_settings",
    "api_key",
    "endpoint",
}


def _patch_list_slot_types() -> bool:
    """Repair `Tool._slot_type_to_python_type` for parameterized list slots.

    FINDING (arklex, not the platform): the mapping covers the bare string
    "list" but not "list[str]", so every parameterized-list slot falls through
    to `Any`. Pydantic renders `Any` as an empty schema `{}`, and OpenAI's
    function-schema validator rejects it with

        Invalid schema for function 'shopify_get_order_details':
        In context=('properties','order_ids','anyOf','0'),
        schema must have a 'type' key.

    Because that 400 rejects the WHOLE tools array, an agent carrying any such
    tool can make no tool calls at all. Three registered tools are affected:
    shopify/get-order-details, shopify/get-products, shopify/cart-add-items,
    and all three are in the shipped Shopify assistant.

    Without this repair the benchmark measures only the crash, so the fix is
    applied here and recorded as a deviation. The upstream fix is the same
    three lines inside `_slot_type_to_python_type`.
    """
    from arklex.resources.tools.tools import Tool

    if getattr(Tool, "_dystopic_list_slot_patch", False):
        return False
    original = Tool._slot_type_to_python_type
    inner_types: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}

    def patched(self: Any, type_str: str) -> Any:
        if isinstance(type_str, str) and type_str.startswith("list[") and type_str.endswith("]"):
            return list[inner_types.get(type_str[5:-1].strip(), str)]
        return original(self, type_str)

    Tool._slot_type_to_python_type = patched
    Tool._dystopic_list_slot_patch = True
    return True


def _install_proxy_tools(allowed_slugs: set[str] | None = None) -> list[str]:
    """Replace each Shopify tool's `func` with a proxy shim.

    `RESOURCE_MAP[slug]["item_cls"]` holds the module-level `Tool` object built
    by `@register_tool` at import time; `ResourceLoader` later does
    `base_tool.copy()`, which carries `self.func` through. Swapping `.func`
    here therefore reaches every instance the executor builds.

    ``allowed_slugs`` restricts patching to the configuration's resolved tool
    surface. Tools outside it are left un-shimmed *and* absent from the
    taskgraph, so they are genuinely unreachable in that configuration rather
    than merely un-advertised — if the agent somehow reached one it would hit
    the real Shopify client and fail, which is the honest outcome.
    """
    from arklex.resources.resource_map import RESOURCE_MAP

    patched: list[str] = []
    for slug, (tool_name, renderer, out_field) in TOOL_SPECS.items():
        if allowed_slugs is not None and slug not in allowed_slugs:
            continue
        entry = RESOURCE_MAP.get(slug)
        if entry is None:
            _CALLS.append({"tool": tool_name, "outcome": "not_in_resource_map", "error": slug})
            continue
        tool_obj = entry["item_cls"]
        original = tool_obj.func
        output_model = original.__annotations__.get("return")

        def _shim(
            *,
            _tool_name: str = tool_name,
            _renderer: Any = renderer,
            _out_field: str = out_field,
            _model: Any = output_model,
            **kwargs: Any,
        ) -> Any:
            args = {
                k: v
                for k, v in kwargs.items()
                if k not in _NON_TOOL_KWARGS and v is not None and v != ""
            }
            try:
                resp = proxy_call_with(_ENVELOPE, _tool_name, args)
                if not isinstance(resp, dict):
                    resp = {"value": resp}
                text = _renderer(resp)
                _CALLS.append({"tool": _tool_name, "outcome": "ok", "args": sorted(args)})
            except Exception as exc:  # noqa: BLE001 - recorded, then surfaced
                _CALLS.append(
                    {"tool": _tool_name, "outcome": "error",
                     "error": f"{type(exc).__name__}: {exc}"}
                )
                text = f"The {_tool_name} call did not succeed: {exc}"
            if _model is not None:
                try:
                    return _model(**{_out_field: text})
                except Exception:  # noqa: BLE001 - fall back to the raw text
                    pass
            return text

        tool_obj.func = functools.wraps(original)(_shim)
        patched.append(tool_name)
    return patched


class _ErrorCapture(logging.Handler):
    """Collect WARNING+ log records so failures reach the trace.

    `OpenAIAgent.response` catches every exception, logs it, and returns the
    generic "An error occurred while processing your request." string. Proxy
    topology captures no stdout, so without this handler a 401, a tool crash and
    a model refusal are indistinguishable from outside the sandbox.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg += " | " + "".join(traceback.format_exception(*record.exc_info))[-1200:]
            self.records.append(f"{record.levelname} {record.name}: {msg[:1500]}")
        except Exception:  # noqa: BLE001 - a logging handler must never raise
            pass


def _run_coroutine(coro: Any) -> Any:
    """Drive a coroutine to completion from either a sync or async caller.

    The platform dispatches `run()` from inside an already-running event loop,
    so `asyncio.run` raises "cannot be called from a running event loop". When
    a loop is already running we execute the coroutine on its own loop in a
    worker thread. That is safe here precisely because the proxy envelope is a
    module-level global rather than a ContextVar.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _user_text(task_input: Any) -> str:
    if isinstance(task_input, str):
        return task_input
    if isinstance(task_input, dict):
        for key in ("user_instruction", "instruction", "text", "message", "task", "prompt", "input"):
            val = task_input.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return json.dumps(task_input) if task_input else ""


# The wire names the frozen harness variant has travelled under, newest first.
# `configuration` is the pre-rename name and is kept purely so a payload from
# either side of the platform's rename resolves; see `_frozen_variant_block`.
_VARIANT_KEYS = ("harness_variant", "configuration")


def _frozen_variant_block(task_input: Any) -> dict[str, Any] | None:
    """The frozen harness variant in *task_input*, or None when none was frozen.

    Returns the FIRST key in ``_VARIANT_KEYS`` that carries a non-empty dict, so
    a platform sending both the current name and the deprecated alias resolves
    to the current one and the two can never be read as two different variants.

    None is a legitimate answer meaning "this check froze no variant" — the
    platform omits the key entirely on that path rather than sending null, to
    keep the pre-variant payload byte-identical. That is exactly why this
    function must try every name it has ever been called: an absent key and a
    misspelled key are indistinguishable here, and guessing wrong silently
    downgrades the run to the static taskgraph instead of failing it.
    """
    if not isinstance(task_input, dict):
        return None
    for key in _VARIANT_KEYS:
        block = task_input.get(key)
        if isinstance(block, dict) and block:
            return block
    return None


def run(task_input: Any = None, *, proxy_url: str | None = None,
        run_token: str | None = None, **_: Any) -> dict[str, Any]:
    global _ENVELOPE
    _CALLS.clear()

    if isinstance(task_input, dict):
        task_input_map = task_input
    else:
        task_input_map = {"input": task_input}
    user_text = _user_text(task_input)

    _ENVELOPE = Envelope(
        proxy_url=proxy_url or os.environ.get("DYSTOPIC_PROXY_URL", ""),
        run_token=run_token or os.environ.get("DYSTOPIC_RUN_TOKEN", ""),
        user_instruction=user_text,
        task_input=task_input_map,
    )

    # The framework builds a langchain/OpenAI client at import of ModelService.
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "sk-unused"))

    slot_patch = _patch_list_slot_types()

    # ---- THE HARNESS-VARIANT SEAM -------------------------------------
    # `harness_variant` is the frozen harness variant the platform is grading
    # this run under. When the check froze no variant the key is absent, and
    # this port behaves exactly as it did before variants existed: the static
    # taskgraph, all nine tools. That no-variant path is deliberately
    # byte-identical, because it is what every pre-variant check replays.
    #
    # A configuration we cannot build is a HARD failure, never a fallback to
    # defaults: silently running the default agent under a variant's name would
    # report a green suite for a configuration that never actually ran.
    #
    # BOTH NAMES ARE READ, new one first. The platform renamed this key from
    # `configuration` to `harness_variant` after this port was written, and
    # because an absent key legitimately means "no variant was frozen" the
    # rename did not error — it silently took the static-taskgraph branch below
    # and registered all nine tools under a five-tool variant's name. Reading
    # the alias costs nothing and closes that failure for a payload sent by
    # either side of the rename.
    configuration = _frozen_variant_block(task_input_map)
    config_meta: dict[str, Any] = {"source": "static_taskgraph"}
    allowed_slugs: set[str] | None = None

    if isinstance(configuration, dict) and configuration:
        knob_values = configuration.get("knob_values") or {}
        resolved, defaulted = load_configuration(knob_values)
        surface = tool_surface(resolved)
        allowed_slugs = {slug for slug, _, _ in surface}
        config = build_taskgraph(resolved)
        config_meta = {
            "source": "harness_variant",
            "variant_name": configuration.get("name"),
            "variant_id": configuration.get("variant_id"),
            "snapshot_id": configuration.get("snapshot_id"),
            "fingerprint": configuration.get("fingerprint"),
            "knob_values": resolved,
            "defaulted_knobs": defaulted,
            "tool_surface": sorted(fn for _, fn, _ in surface),
        }
    else:
        config = json.loads(TASKGRAPH.read_text())

    patched = _install_proxy_tools(allowed_slugs)

    capture = _ErrorCapture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.WARNING)

    from arklex.models.llm_config import LLMConfig
    from arklex.orchestrator.executor.executor import Executor
    from arklex.orchestrator.orchestrator import AgentOrg

    llm_config = LLMConfig.model_validate(config["llm_config"])
    config["model"] = config["llm_config"]

    final_response = ""
    error: str | None = None
    try:
        executor = Executor(
            tools=config.get("tools", []),
            workers=config.get("workers", []),
            nodes=config.get("nodes", []),
            llm_config=llm_config,
        )
        orchestrator = AgentOrg(config=config, executor=executor)
        # get_response is `async def`; the openai-agent node runs the OpenAI
        # Agents SDK Runner on that loop, and Tool.execute dispatches sync tool
        # bodies through asyncio.to_thread, which is why the proxy envelope is
        # a module-level global rather than a ContextVar.
        result = _run_coroutine(
            orchestrator.get_response(
                {"text": user_text, "chat_history": [], "parameters": {}}
            )
        )
        if isinstance(result, dict):
            final_response = (
                result.get("answer")
                or result.get("response")
                or result.get("final_response")
                or ""
            )
            if not final_response:
                final_response = json.dumps(result)[:4000]
        else:
            final_response = str(result)

        # Guard: a stringified Python object is never a real answer. The first
        # run of this port shipped "<coroutine object AgentOrg.get_response ...>"
        # as final_response and was graded as eight ordinary agent failures.
        if re.match(r"^<[\w.]+ object at 0x[0-9a-f]+>$", final_response.strip()) or \
                final_response.strip().startswith("<coroutine object"):
            raise RuntimeError(
                f"entrypoint produced a stringified object, not an answer: {final_response[:120]}"
            )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        error = f"{type(exc).__name__}: {exc}"
        final_response = f"The assistant failed to complete the request: {error}"
        _CALLS.append({"tool": "__entrypoint__", "outcome": "error",
                       "error": traceback.format_exc()[-1500:]})
    finally:
        logging.getLogger().removeHandler(capture)

    return {
        "final_response": final_response or "(no response produced)",
        "metadata": {
            # THE ACKNOWLEDGEMENT. The platform compares this against the
            # snapshot it froze and fails the run when they disagree — a harness
            # that ran the static taskgraph under a variant's name is a contract
            # breach, not a scenario failure. Named for the wire key it answers.
            "harness_variant": config_meta,
            "patched_tools": patched,
            "list_slot_type_patch_applied": slot_patch,
            "tool_calls": _CALLS,
            "tool_call_count": len([c for c in _CALLS if c.get("outcome") == "ok"]),
            "error": error,
            "log_errors": capture.records[-12:],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run({"user_instruction": " ".join(sys.argv[1:])}), indent=2)[:4000])
