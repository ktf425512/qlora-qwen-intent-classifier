"""
Label vocabulary and post-processing shared by evaluation and inference.

The 27 intents are the Bitext customer-support taxonomy used in train.py.
`match_label` maps a raw generation to one of them, or to "__other__" when the
model produced something outside the vocabulary (label hallucination). It is the
same rule train.py applies at evaluation time, so eval and inference agree on
what counts as a valid prediction.
"""
from __future__ import annotations

OTHER = "__other__"

VALID_LABELS: tuple[str, ...] = (
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
)


def match_label(pred: str | None, valid=VALID_LABELS) -> str:
    """Map a raw generation string to the closest valid label (or '__other__').

    Rules, in order: exact match on the first whitespace-delimited token
    (case-insensitive), then any valid label appearing as a substring of the
    generation, otherwise '__other__'.
    """
    if not pred or not pred.strip():
        return OTHER
    p = pred.strip().lower().split()[0]
    for label in valid:
        if p == label.lower():
            return label
    for label in valid:
        if label.lower() in pred.lower():
            return label
    return OTHER
