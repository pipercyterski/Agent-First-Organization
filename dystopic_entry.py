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
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _install_proxy_tools() -> list[str]:
    """Replace each Shopify tool's `func` with a proxy shim.

    `RESOURCE_MAP[slug]["item_cls"]` holds the module-level `Tool` object built
    by `@register_tool` at import time; `ResourceLoader` later does
    `base_tool.copy()`, which carries `self.func` through. Swapping `.func`
    here therefore reaches every instance the executor builds.
    """
    from arklex.resources.resource_map import RESOURCE_MAP

    patched: list[str] = []
    for slug, (tool_name, renderer, out_field) in TOOL_SPECS.items():
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


def _user_text(task_input: Any) -> str:
    if isinstance(task_input, str):
        return task_input
    if isinstance(task_input, dict):
        for key in ("user_instruction", "instruction", "text", "message", "task", "prompt", "input"):
            val = task_input.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return json.dumps(task_input) if task_input else ""


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

    patched = _install_proxy_tools()

    from arklex.models.llm_config import LLMConfig
    from arklex.orchestrator.executor.executor import Executor
    from arklex.orchestrator.orchestrator import AgentOrg

    config = json.loads(TASKGRAPH.read_text())
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
        # Agents SDK Runner on this loop, and Tool.execute dispatches sync tool
        # bodies through asyncio.to_thread, which is why the proxy envelope is
        # a module-level global rather than a ContextVar.
        result = asyncio.run(
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

    return {
        "final_response": final_response or "(no response produced)",
        "metadata": {
            "patched_tools": patched,
            "tool_calls": _CALLS,
            "tool_call_count": len([c for c in _CALLS if c.get("outcome") == "ok"]),
            "error": error,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run({"user_instruction": " ".join(sys.argv[1:])}), indent=2)[:4000])
