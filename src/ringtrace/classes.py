"""The twelve labels, in the organisers' order. Every array in the pipeline uses this order."""

CLASSES: tuple[str, ...] = (
    "legit",
    "mule_account",
    "velocity_bot",
    "layering_chain",
    "sim_swap",
    "account_takeover",
    "phishing_collect",
    "qr_swap",
    "refund_abuse",
    "fake_cashback",
    "loan_app_extortion",
    "merchant_collusion",
)
N_CLASSES = len(CLASSES)
CODE = {name: i for i, name in enumerate(CLASSES)}
LEGIT = 0
FRAUD_CODES = tuple(range(1, N_CLASSES))
