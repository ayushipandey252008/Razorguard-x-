ROLES = ("ADMIN", "RISK_ANALYST", "INVESTIGATOR", "VIEWER")

ROLE_LEVEL = {
    "VIEWER": 1,
    "INVESTIGATOR": 2,
    "RISK_ANALYST": 3,
    "ADMIN": 4,
}


def has_role(user_role: str, required: str | list[str]) -> bool:
    if isinstance(required, str):
        required = [required]
    return user_role in required or user_role == "ADMIN"


def can_decide(role: str) -> bool:
    return ROLE_LEVEL.get(role, 0) >= ROLE_LEVEL["RISK_ANALYST"]


def can_investigate(role: str) -> bool:
    return ROLE_LEVEL.get(role, 0) >= ROLE_LEVEL["INVESTIGATOR"]
