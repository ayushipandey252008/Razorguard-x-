from __future__ import annotations

from app.rules.engine import RuleEngine, RuleResult

engine = RuleEngine()


@engine.register
def high_amount(txn: dict, ctx: dict) -> RuleResult | None:
    amount = float(txn["amount"])
    baseline = float(ctx.get("typical_amount") or txn.get("previous_average_amount") or 0)
    triggered = amount >= 50_000 or (baseline > 0 and amount >= 5 * baseline)
    return RuleResult(
        rule_id="HIGH_AMOUNT",
        rule_name="Unusually high transaction amount",
        severity="high" if amount >= 50_000 else "medium",
        score_contribution=22 if triggered else 0,
        explanation=f"Amount {amount:.2f} vs baseline {baseline:.2f}",
        evidence={"amount": amount, "baseline": baseline},
        triggered=triggered,
    )


@engine.register
def new_device(txn: dict, ctx: dict) -> RuleResult | None:
    known = bool(txn.get("current_device_known", True))
    return RuleResult(
        rule_id="NEW_DEVICE",
        rule_name="New or unknown device",
        severity="medium",
        score_contribution=12 if not known else 0,
        explanation="Transaction originated from a device not previously associated with this user",
        evidence={"device_id": txn.get("device_id"), "current_device_known": known},
        triggered=not known,
    )


@engine.register
def unusual_location(txn: dict, ctx: dict) -> RuleResult | None:
    known = bool(txn.get("current_location_known", True))
    return RuleResult(
        rule_id="UNUSUAL_LOCATION",
        rule_name="Unusual location",
        severity="medium",
        score_contribution=12 if not known else 0,
        explanation="Location is not in the user's known location set",
        evidence={"location": txn.get("location"), "current_location_known": known},
        triggered=not known,
    )


@engine.register
def velocity(txn: dict, ctx: dict) -> RuleResult | None:
    vel = int(txn.get("transaction_velocity") or 1)
    triggered = vel >= 6
    return RuleResult(
        rule_id="HIGH_VELOCITY",
        rule_name="Excessive transaction velocity",
        severity="high" if vel >= 10 else "medium",
        score_contribution=18 if vel >= 10 else (12 if triggered else 0),
        explanation=f"{vel} transactions observed in the recent velocity window",
        evidence={"transaction_velocity": vel},
        triggered=triggered,
    )


@engine.register
def failed_attempts(txn: dict, ctx: dict) -> RuleResult | None:
    fails = int(txn.get("failed_attempts") or 0)
    triggered = fails >= 3
    return RuleResult(
        rule_id="FAILED_ATTEMPTS",
        rule_name="Repeated failed attempts",
        severity="medium",
        score_contribution=10 if triggered else 0,
        explanation=f"{fails} failed attempts preceded this transaction",
        evidence={"failed_attempts": fails},
        triggered=triggered,
    )


@engine.register
def device_reuse(txn: dict, ctx: dict) -> RuleResult | None:
    users = ctx.get("device_user_count") or 1
    triggered = users >= 3
    return RuleResult(
        rule_id="DEVICE_REUSE",
        rule_name="Suspicious device reuse",
        severity="high",
        score_contribution=16 if triggered else 0,
        explanation=f"Device is associated with {users} users",
        evidence={"device_id": txn.get("device_id"), "distinct_users": users},
        triggered=triggered,
    )


@engine.register
def ip_reuse(txn: dict, ctx: dict) -> RuleResult | None:
    users = ctx.get("ip_user_count") or 1
    triggered = users >= 3
    return RuleResult(
        rule_id="IP_REUSE",
        rule_name="Suspicious IP reuse",
        severity="high",
        score_contribution=16 if triggered else 0,
        explanation=f"IP is associated with {users} users",
        evidence={"ip_address": txn.get("ip_address"), "distinct_users": users},
        triggered=triggered,
    )


@engine.register
def watchlisted_merchant(txn: dict, ctx: dict) -> RuleResult | None:
    flagged = bool(ctx.get("merchant_watchlisted"))
    return RuleResult(
        rule_id="WATCHLISTED_MERCHANT",
        rule_name="Previously flagged merchant",
        severity="high",
        score_contribution=20 if flagged else 0,
        explanation="Merchant is on the prototype watchlist",
        evidence={"merchant_id": txn.get("merchant_id")},
        triggered=flagged,
    )


@engine.register
def new_account_high_value(txn: dict, ctx: dict) -> RuleResult | None:
    age = int(txn.get("account_age_days") or 0)
    amount = float(txn["amount"])
    triggered = age < 7 and amount >= 10_000
    return RuleResult(
        rule_id="NEW_ACCOUNT_HIGH_VALUE",
        rule_name="New account high-value payment",
        severity="high",
        score_contribution=18 if triggered else 0,
        explanation=f"Account age {age} days with amount {amount:.2f}",
        evidence={"account_age_days": age, "amount": amount},
        triggered=triggered,
    )


@engine.register
def behavior_deviation(txn: dict, ctx: dict) -> RuleResult | None:
    anomalies = ctx.get("anomalies") or []
    codes = {a.get("code") for a in anomalies}
    triggered = "amount_deviation" in codes and (
        "unknown_device" in codes or "unknown_location" in codes
    )
    return RuleResult(
        rule_id="BEHAVIOR_DEVIATION",
        rule_name="Account behavior deviation",
        severity="high",
        score_contribution=14 if triggered else 0,
        explanation="Amount deviation combined with unknown device or location",
        evidence={"anomaly_codes": list(codes)},
        triggered=triggered,
    )
