"""The harness's own configuration loader — the object under test.

WHY THIS FILE EXISTS
--------------------
arklex's product is an agent *builder*: an end user writes a config, the
framework compiles it into a task graph, and the resulting agent has whatever
tool set that config asked for. Two customers of arklex therefore run two
different agents out of one repo at one commit.

Dystopic models that with a **harness variant** — one named configuration, a
flat ``{knob: value}`` map. The platform hands the frozen configuration to this
entrypoint in ``task_input["configuration"]["knob_values"]``; this module is the
adapter that maps those knob values onto the repo's real taskgraph constructor.

DESIGN RULES (from docs/plans/platform-customer-harness-variants.plan.md §2)
---------------------------------------------------------------------------
1. **Do not inject past the loader.** Config-loading is itself under test, so
   the knob values go in at the top (``load_configuration``) and the taskgraph
   is built from them by the same code path a real arklex user's config would
   take. A harness that mis-parses its own knobs must FAIL the check here
   rather than be papered over by the platform.
2. **Fail closed on an unknown or ill-typed knob.** Silently ignoring a knob
   would make a configuration that the platform believes is under test into one
   that never actually varied — the worst failure mode this feature has,
   because every suite would still go green.
3. **No defaults invented for declared knobs.** A knob the platform declared but
   did not send is a real authoring bug; it resolves to the documented default
   below and is reported in ``metadata.configuration.defaulted_knobs`` so the
   trace shows it rather than hiding it.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# The knob space this harness declares.
#
# Mirrors `agents.harness_schema` on the platform side. Keeping the two in
# sync is the port's job; phase 2 of the plan adds a `--dry-run` diff between
# them, and any drift is itself a finding.
# --------------------------------------------------------------------------
KNOB_SPEC: dict[str, dict[str, Any]] = {
    "mutations_enabled": {
        "type": bool,
        "default": True,
        "description": "Expose the write tools (cancel_order, return_products, cart_add_items).",
    },
    "cart_enabled": {
        "type": bool,
        "default": True,
        "description": "Expose the cart surface (get_cart, and cart_add_items when mutations are on).",
    },
    "catalog_search_enabled": {
        "type": bool,
        "default": True,
        "description": "Expose catalog browsing (search_products, get_products, get_web_product).",
    },
    "escalation_policy": {
        "type": str,
        "default": "self_serve",
        "enum": ("self_serve", "confirm_first"),
        "description": "self_serve acts directly; confirm_first must state the action and ask before any write.",
    },
    "agent_model": {
        "type": str,
        "default": "gpt-4o",
        "enum": ("gpt-4o", "gpt-4o-mini"),
        "description": "Model backing the openai-agent node.",
    },
}

# (ToolItem slug, registered function name, task description, the knob predicate)
#
# The predicate is what makes this a harness rather than one agent: the tool
# surface is a function of the configuration, so two variants at the same commit
# genuinely call different tools.
TOOL_CATALOG: list[tuple[str, str, str, Any]] = [
    ("shopify/get-user-details-admin", "get_user_details_admin",
     "Look up the customer's account record by their customer id",
     lambda c: True),
    ("shopify/get-order-details", "get_order_details",
     "List the customer's orders with status, totals, cancellability and line items",
     lambda c: True),
    ("shopify/search-products", "search_products",
     "Search the catalog for products matching a free-text query",
     lambda c: c["catalog_search_enabled"]),
    ("shopify/get-products", "get_products",
     "Fetch catalog products by their product ids",
     lambda c: c["catalog_search_enabled"]),
    ("shopify/get-web-product", "get_web_product",
     "Fetch a single storefront product by id",
     lambda c: c["catalog_search_enabled"]),
    ("shopify/get-cart", "get_cart",
     "Read the contents of the customer's cart",
     lambda c: c["cart_enabled"]),
    ("shopify/cart-add-items", "cart_add_items",
     "Add product variants to the customer's cart",
     lambda c: c["cart_enabled"] and c["mutations_enabled"]),
    ("shopify/return-products", "return_products",
     "Submit a return request for a delivered order",
     lambda c: c["mutations_enabled"]),
    ("shopify/cancel-order", "cancel_order",
     "Cancel an order, which notifies the customer, restocks inventory and refunds payment",
     lambda c: c["mutations_enabled"]),
]

ROLE = "customer service assistant"
USER_OBJECTIVE = (
    "The customer service assistant helps users with customer service inquiries. "
    "It can provide information about products, services, and policies, as well as "
    "help users resolve issues and complete transactions."
)
INTRO = (
    "Sweet for everyone is an online retailer offering a curated selection of "
    "children's apparel and home accessories. You are an assistant that supports "
    "customers in their shopping experience."
)

# Per-tool prompt lines, so the agent's instructions describe the surface it
# actually has. A prompt advertising a tool the configuration removed is the
# single most common way a config-driven harness degrades: the model keeps
# trying to call a tool that is not there and burns the turn.
_TOOL_PROMPT_LINES: dict[str, str] = {
    "get_user_details_admin": "- get_user_details_admin: look up a customer account by user_id.",
    "get_order_details": (
        "- get_order_details: list a customer's orders. Each order carries its status,\n"
        "  fulfillment_status, return_status, cancellable and cancel_block_reason."
    ),
    "search_products": "- search_products: search the catalog by free text.",
    "get_products": "- get_products: fetch catalog products by product id.",
    "get_web_product": "- get_web_product: fetch a single storefront product by id.",
    "get_cart": "- get_cart: read the customer's cart.",
    "cart_add_items": "- cart_add_items: add product variants to the customer's cart.",
    "return_products": "- return_products: submit a return request for an order.",
    "cancel_order": (
        "- cancel_order: cancel an order. This notifies the customer, restocks the\n"
        "  inventory and refunds the payment. It cannot be undone."
    ),
}

_ESCALATION_PROMPT: dict[str, str] = {
    "self_serve": (
        "You may take actions the customer asks for directly, once you have checked "
        "the record supports them."
    ),
    "confirm_first": (
        "Before ANY action that changes an order, a return or a cart, you must first "
        "state exactly what you are about to do and ask the customer to confirm. Do "
        "not call a write tool in the same turn as the confirmation request."
    ),
}


class ConfigurationError(ValueError):
    """The configuration this harness was handed is not one it can build.

    Raised, not swallowed: an unbuildable configuration must fail the check
    loudly. A harness that quietly falls back to its defaults would report a
    green suite for a configuration that never ran.
    """


def load_configuration(knob_values: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate + resolve the platform's knob values into this harness's config.

    Returns ``(config, defaulted)`` where ``defaulted`` names the declared knobs
    the platform did not send. Raises :class:`ConfigurationError` on an unknown
    knob, a wrong type, or a value outside a declared enum.
    """
    if knob_values is None:
        knob_values = {}
    if not isinstance(knob_values, dict):
        raise ConfigurationError(
            f"knob_values must be a JSON object, got {type(knob_values).__name__}"
        )

    unknown = sorted(set(knob_values) - set(KNOB_SPEC))
    if unknown:
        raise ConfigurationError(
            f"unknown knob(s) for this harness: {', '.join(unknown)}. "
            f"Declared knobs are: {', '.join(sorted(KNOB_SPEC))}"
        )

    config: dict[str, Any] = {}
    defaulted: list[str] = []
    for knob, spec in KNOB_SPEC.items():
        if knob not in knob_values:
            config[knob] = spec["default"]
            defaulted.append(knob)
            continue
        value = knob_values[knob]
        expected = spec["type"]
        # bool is a subclass of int; check it first and exactly, or an int knob
        # would silently accept True.
        if expected is bool:
            if not isinstance(value, bool):
                raise ConfigurationError(
                    f"knob {knob!r} must be a boolean, got {type(value).__name__}: {value!r}"
                )
        elif not isinstance(value, expected) or isinstance(value, bool):
            raise ConfigurationError(
                f"knob {knob!r} must be {expected.__name__}, got {type(value).__name__}: {value!r}"
            )
        if "enum" in spec and value not in spec["enum"]:
            raise ConfigurationError(
                f"knob {knob!r} must be one of {list(spec['enum'])}, got {value!r}"
            )
        config[knob] = value

    return config, defaulted


def tool_surface(config: dict[str, Any]) -> list[tuple[str, str, str]]:
    """The (slug, function name, task) triples this configuration exposes."""
    return [
        (slug, fn, task)
        for slug, fn, task, predicate in TOOL_CATALOG
        if predicate(config)
    ]


def agent_prompt(config: dict[str, Any], surface: list[tuple[str, str, str]]) -> str:
    """The openai-agent node prompt, describing only the resolved surface."""
    lines = [_TOOL_PROMPT_LINES[fn] for _, fn, _ in surface if fn in _TOOL_PROMPT_LINES]
    tools_block = "\n".join(lines) if lines else "- (none — you have no tools in this configuration)"
    return "\n".join(
        [
            INTRO,
            "",
            f"Your role: {ROLE}. {USER_OBJECTIVE}",
            "",
            "You have these tools:",
            tools_block,
            "",
            _ESCALATION_PROMPT[config["escalation_policy"]],
            "",
            "Check the order's own fields before you act on it. Tell the customer plainly",
            "what you did or could not do. If the customer asks for something none of your",
            "tools can do, say so directly instead of implying it was done.",
        ]
    ).strip()


def build_taskgraph(config: dict[str, Any]) -> dict[str, Any]:
    """Compile a configuration into the taskgraph the framework loads.

    Same shape ``dystopic/build_taskgraph.py`` writes for the static port — the
    only difference is that the tool list, the prompt and the model come from
    the configuration instead of being fixed.
    """
    surface = tool_surface(config)
    nodes: list[Any] = [
        [
            "0",
            {
                "resource": {"id": "openai-agent"},
                "attribute": {"task": "", "type": "agent", "start": True, "can_skipped": False},
                "data": {
                    "name": "Sweet For Everyone Support",
                    "prompt": agent_prompt(config, surface),
                    "agent_start_message": "Hi! How can I help you with your order today?",
                },
            },
        ]
    ]
    edges: list[Any] = []
    for i, (slug, fn_name, task) in enumerate(surface, start=1):
        nodes.append(
            [
                str(i),
                {
                    "resource": {"id": slug},
                    "attribute": {"task": task, "type": "tool", "start": False, "can_skipped": False},
                    "data": {"name": fn_name},
                },
            ]
        )
        edges.append(
            [
                "0",
                str(i),
                {
                    "intent": "None",
                    "attribute": {"weight": 1, "pred": True, "definition": "", "sample_utterances": []},
                },
            ]
        )

    return {
        "llm_config": {
            "model_type_or_path": config["agent_model"],
            "llm_provider": "openai",
            "langchain_model_kwargs": {},
        },
        "nodes": nodes,
        "edges": edges,
        "workers": [],
        "tools": [{"id": slug} for slug, _, _ in surface],
        "agents": [{"id": "openai-agent"}],
        "role": ROLE,
        "user_objective": USER_OBJECTIVE,
        "domain": "Shopify sellers",
        "intro": INTRO,
    }
