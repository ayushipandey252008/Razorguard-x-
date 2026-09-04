from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnomalySignal:
    code: str
    description: str
    severity: str
    contribution: float


def analyze_behavior(txn: dict, user: dict | None, iso_score: float) -> dict:
    """Personalized explainable checks blended with a **global** Isolation Forest.

    Global: Isolation Forest decision, mapped to 0–100 (`anomaly_score`).
    Personalized: amount vs that user's typical, hour vs typical hour,
    known device/location, velocity vs a mild per-user expectation.
    """
    signals: list[AnomalySignal] = []
    amount = float(txn["amount"])
    typical = float((user or {}).get("typical_amount") or txn.get("previous_average_amount") or 0)
    hour = txn["timestamp"].hour if hasattr(txn.get("timestamp"), "hour") else 12
    typical_hour = int((user or {}).get("typical_hour") or 14)
    prev_count = int(txn.get("previous_transaction_count") or 0)

    amount_z = None
    if typical > 0:
        scale = max(typical * 0.35, 1.0)
        amount_z = (amount - typical) / scale
        ratio = amount / typical
        if ratio >= 4 or (amount_z is not None and amount_z >= 3.5):
            signals.append(
                AnomalySignal(
                    "amount_deviation",
                    f"Amount {amount:.2f} is {ratio:.1f}x user typical {typical:.2f} (z≈{amount_z:.1f})",
                    "high",
                    28,
                )
            )
        elif ratio >= 2 or (amount_z is not None and amount_z >= 2.0):
            signals.append(
                AnomalySignal(
                    "amount_deviation",
                    f"Amount {amount:.2f} is {ratio:.1f}x user typical {typical:.2f} (z≈{amount_z:.1f})",
                    "medium",
                    14,
                )
            )

    if not txn.get("current_device_known", True):
        signals.append(AnomalySignal("unknown_device", "Device is not in the user's known-device set", "high", 18))
    if not txn.get("current_location_known", True):
        signals.append(
            AnomalySignal("unknown_location", "Location is not in the user's known-location set", "medium", 14)
        )

    hour_delta = min(abs(hour - typical_hour), 24 - abs(hour - typical_hour))
    if hour_delta >= 8:
        signals.append(
            AnomalySignal(
                "unusual_hour",
                f"Hour {hour}:00 is {hour_delta}h from typical hour {typical_hour}:00",
                "low",
                8,
            )
        )

    velocity = int(txn.get("transaction_velocity") or 1)
    # Established accounts rarely need velocity > 4; new accounts still use absolute cuts.
    vel_cut_high = 8 if prev_count >= 10 else 6
    vel_cut_med = 5 if prev_count >= 10 else 4
    if velocity >= vel_cut_high:
        signals.append(
            AnomalySignal("high_velocity", f"Velocity {velocity} in the recent window (user history n={prev_count})", "high", 22)
        )
    elif velocity >= vel_cut_med:
        signals.append(
            AnomalySignal("elevated_velocity", f"Velocity {velocity} in the recent window (user history n={prev_count})", "medium", 12)
        )

    personalized = min(100.0, sum(s.contribution for s in signals))
    # Isolation Forest remains a global novelty score; it is not user-conditional.
    behavior_score = round(0.35 * iso_score + 0.65 * personalized, 2)
    return {
        "anomaly_score": round(iso_score, 2),
        "behavior_score": min(100.0, behavior_score),
        "isolation_forest_scope": "global",
        "personalized_scope": [
            "amount_vs_user_typical",
            "hour_vs_user_typical",
            "device_familiarity",
            "location_familiarity",
            "velocity_vs_history_size",
        ],
        "amount_z": None if amount_z is None else round(float(amount_z), 3),
        "detected_anomalies": [
            {
                "code": s.code,
                "description": s.description,
                "severity": s.severity,
                "contribution": s.contribution,
            }
            for s in signals
        ],
        "user_baseline": {
            "typical_amount": typical,
            "typical_hour": typical_hour,
            "known_devices": (user or {}).get("known_devices") or [],
            "known_locations": (user or {}).get("known_locations") or [],
            "home_location": (user or {}).get("home_location"),
        },
    }
