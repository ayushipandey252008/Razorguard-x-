from __future__ import annotations


class DatasetValidationError(ValueError):
    """Raised when the ULB CSV cannot be used. Fail loudly; do not train."""
