"""
tools.py
--------
This file defines all the "tools" (functions) that the AI agent can call
to interact with the real world: reading/writing the live inventory in
Google Sheets, searching the FAQ knowledge base, and processing orders.

Each tool returns a structured dictionary in the shape:
    {"success": bool, "data": ..., "message": str}
instead of a raw string. This makes the tool output predictable and easy
for the LLM to parse correctly, which significantly reduces hallucination
compared to returning free-form text.
"""

from sheets_connection import get_sheet
from rag_setup import db
from langchain_core.tools import tool
from pydantic import BaseModel


class OrderItem(BaseModel):
    """
    Defines the exact shape of a single item inside an order.
    Using Pydantic here guarantees the LLM always sends 'flavor' and
    'quantity' with consistent field names and types -- this eliminates
    an entire class of bugs (e.g. "Quantity" vs "quantity" mismatches)
    that would otherwise happen if we accepted a plain dict.
    """
    flavor: str
    quantity: int


# ---------------------------------------------------------------------
# READ-ONLY TOOLS
# These tools only look up information; they never modify the inventory.
# ---------------------------------------------------------------------

@tool
def get_donut_info(flavor: str):
    """
    Use this tool whenever the customer asks about ONE specific donut
    flavor (price, availability, or remaining quantity).

    Do NOT use this tool if the customer asks about all flavors, or
    about more than one flavor at once -- use get_available_flavors
    instead for those cases.

    Args:
        flavor: The flavor name the customer is asking about.

    Returns:
        A dict with the flavor's price, quantity, and availability,
        or a "not found" message if the flavor doesn't exist.
    """
    try:
        sheet = get_sheet()
        data = sheet.get_all_records()

        for record in data:
            if record["flavor"].lower() == flavor.lower():
                price_number = int(record["price"].split()[0])
                quantity = record["quantity"]
                return {
                    "success": True,
                    "data": {
                        "flavor": record["flavor"],
                        "price": price_number,
                        "quantity": quantity,
                        "available": quantity > 0,
                    },
                }

        return {
            "success": False,
            "data": None,
            "message": f"Sorry, {flavor} was not found",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def get_available_flavors():
    """
    Use this tool whenever the customer asks about the full menu or
    all available flavors in general (e.g. "عندكم إيه؟", "ايه النكهات؟").

    Do NOT use this for a question about one specific flavor --
    use get_donut_info instead.

    Returns:
        A dict containing a list of every flavor currently in stock
        (quantity > 0), each with its price and remaining quantity.
    """
    try:
        sheet = get_sheet()
        data = sheet.get_all_records()
        available = []

        for record in data:
            quantity = record["quantity"]
            if quantity == 0:
                continue  # skip sold-out flavors

            price_number = int(record["price"].split()[0])
            available.append(
                {
                    "flavor": record["flavor"],
                    "quantity": quantity,
                    "price": price_number,
                }
            )

        if not available:
            return {
                "success": False,
                "data": None,
                "message": "No flavors are currently available.",
            }

        return {"success": True, "data": available}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def search_faq(question: str):
    """
    Use this tool ONLY for questions answered by the shop's general
    FAQ document (business hours, delivery info, payment methods,
    flavor descriptions).

    Do NOT use this tool for placing orders, checking inventory,
    checking prices, or canceling orders -- those have their own tools.

    Args:
        question: The customer's question, used to search the FAQ.

    Returns:
        The most relevant snippets from the FAQ knowledge base (RAG).
    """
    try:
        search_result = db.similarity_search(question, k=3)
        formatted_results = [doc.page_content for doc in search_result]

        return {"success": True, "data": formatted_results}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------
# ORDER TOOLS
# These tools modify the live inventory in Google Sheets.
# ---------------------------------------------------------------------

@tool
def place_multiple_order(items: list[OrderItem]):
    """
    Use this tool whenever the customer wants to purchase one or more
    donut flavors. This single tool handles new orders, adding items,
    removing items, and changing quantities -- always send the FULL
    current order (never just the changed item).

    Uses a two-pass validation strategy:
      Pass 1 checks that every item in the order is available in the
      requested quantity, WITHOUT changing the inventory yet.
      Pass 2 only runs if pass 1 succeeded for everything, and then
      actually updates the sheet.
    This prevents a partial failure from leaving the inventory in an
    inconsistent state (e.g. deducting stock for item 1 and 2, then
    failing on item 3).

    Args:
        items: A list of {"flavor": str, "quantity": int} objects
               representing the customer's full order.

    Returns:
        A dict with the line-item breakdown and grand total on
        success, or a message explaining what went wrong on failure.
    """
    try:
        sheet = get_sheet()
        data = sheet.get_all_records()

        # Build a lookup table once (flavor -> {record, row}) so we
        # don't have to re-scan the whole sheet for every item.
        inventory = {}
        for i, record in enumerate(data):
            inventory[record["flavor"].lower()] = {
                "record": record,
                "row": i + 2,  # +2 accounts for the header row + 0-index
            }

        # --- Pass 1: validate everything first ---
        for item in items:
            record_info = inventory.get(item.flavor.lower())

            if record_info is None:
                return {
                    "success": False,
                    "data": None,
                    "message": f"Sorry, we don't have {item.flavor}.",
                }

            current_quantity = record_info["record"]["quantity"]

            if current_quantity == 0:
                return {
                    "success": False,
                    "data": None,
                    "message": f"Sorry, {record_info['record']['flavor']} is out of stock.",
                }

            if current_quantity < item.quantity:
                return {
                    "success": False,
                    "data": None,
                    "message": (
                        f"Sorry, we only have {current_quantity} left "
                        f"of {record_info['record']['flavor']}."
                    ),
                }

        # --- Pass 2: everything is valid, now actually commit it ---
        results = []
        grand_total = 0

        for item in items:
            record_info = inventory[item.flavor.lower()]
            record = record_info["record"]
            row = record_info["row"]

            unit_price = int(record["price"].split()[0])
            new_quantity = record["quantity"] - item.quantity
            sheet.update_cell(row, 2, new_quantity)

            item_total = unit_price * item.quantity
            grand_total += item_total

            results.append(
                {
                    "flavor": record["flavor"],
                    "quantity": item.quantity,
                    "unit_price": unit_price,
                    "item_total": item_total,
                }
            )

        return {
            "success": True,
            "data": {"items": results, "grand_total": grand_total},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def cancel_order(flavor: str, quantity: int):
    """
    Use this tool whenever the customer wants to cancel part or all
    of their order. Returns the given quantity back to the inventory.

    Args:
        flavor: The flavor to cancel.
        quantity: The number of pieces to return to stock.

    Returns:
        A confirmation dict on success, or an error message if the
        flavor can't be found.
    """
    try:
        sheet = get_sheet()
        data = sheet.get_all_records()

        for i, record in enumerate(data):
            if record["flavor"].lower() == flavor.lower():
                new_quantity = record["quantity"] + quantity
                sheet.update_cell(i + 2, 2, new_quantity)
                return {
                    "success": True,
                    "data": {"flavor": record["flavor"], "quantity": quantity},
                    "message": "Your order has been canceled.",
                }

        return {
            "success": False,
            "data": None,
            "message": f"We can't find {flavor} in the stock.",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------
# Registry the agent uses to look up a tool by the name the LLM chose.
# ---------------------------------------------------------------------
available_functions = {
    "get_donut_info": get_donut_info,
    "get_available_flavors": get_available_flavors,
    "place_multiple_order": place_multiple_order,
    "search_faq": search_faq,
    "cancel_order": cancel_order,
}