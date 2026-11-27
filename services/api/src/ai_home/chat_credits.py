"""Explicit client spending authorization and receipt validation."""
from uuid import UUID, uuid4


def credit_authorization(limit=0, request_id=None):
    if type(limit) is not int or not 0 <= limit <= 1_000_000:
        raise ValueError("El límite debe ser un entero entre 0 y 1000000 créditos.")
    if not limit and request_id is None:
        return {}
    return {"credit_limit": limit, "credit_request_id": str(UUID(str(request_id))) if request_id else str(uuid4())}


def validate_credit_receipt(data, authorization):
    charged = data.get("charged_credits", 0)
    if type(charged) is not int or not 0 <= charged <= authorization.get("credit_limit", 0):
        raise ValueError("El cobro comunicado supera el límite autorizado o es inválido.")
    if charged and data.get("credit_request_id") != authorization.get("credit_request_id"):
        raise ValueError("El recibo de créditos no corresponde a esta solicitud.")
    return charged
