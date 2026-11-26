"""Cancellable desktop-side consumer of the AI@Home SSE protocol."""
import asyncio
import json
from uuid import UUID

import httpx

from ai_home.chat_credits import credit_authorization, validate_credit_receipt
from ai_home.schemas import KnowledgeReference, FoundationGeneration
from ai_home.urls import normalize_base_url


class GenerationCancelled(RuntimeError):
    pass


def stream_server_chat(server_url, user_token, messages, *, cancel, on_delta,
                       model_version=None, use_knowledge=False, generation=None, transport=None, credit_limit=0, credit_request_id=None):
    """Run in a background thread; callbacks must not directly mutate Tk widgets."""
    authorization = credit_authorization(credit_limit, credit_request_id)
    async def consume():
        body = {"messages": messages, "use_knowledge": use_knowledge, **authorization}
        if generation is not None:
            body["generation"] = FoundationGeneration.model_validate(generation).model_dump(exclude_none=True)
        if model_version is not None:
            body["model_version"] = str(UUID(model_version))
        async with httpx.AsyncClient(base_url=normalize_base_url(server_url), timeout=180,
                                     headers={"Authorization": f"Bearer {user_token}"},
                                     transport=transport) as client:
            async with client.stream("POST", "/api/v1/chat/stream", json=body) as response:
                response.raise_for_status()
                if not response.headers.get("content-type", "").startswith("text/event-stream"):
                    raise ValueError("El servidor no devolvió un flujo de texto válido.")
                metadata, pieces = None, []
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        raise ValueError("Evento de streaming inválido.")
                    event = json.loads(line[6:])
                    if not isinstance(event, dict):
                        raise ValueError("Evento de streaming inválido.")
                    kind = event.get("type")
                    if kind == "error":
                        raise RuntimeError({"insufficient_credits": "Saldo insuficiente para esta inferencia.",
                                            "credit_request_conflict": "La solicitud ya fue procesada; no se repitió el cobro."}.get(
                                                event.get("code"), "El servidor no pudo completar la inferencia."))
                    if kind == "metadata":
                        if metadata is not None or pieces:
                            raise ValueError("Metadatos repetidos o fuera de orden.")
                        actual = str(UUID(event["model_version"])) if event.get("model_version") else None
                        if model_version is not None and actual != str(UUID(model_version)):
                            raise ValueError("El servidor respondió con una versión distinta.")
                        metadata = {"model_version": actual, "references": [
                            KnowledgeReference.model_validate(r).model_dump() for r in event.get("references", [])]}
                    elif kind == "delta":
                        if metadata is None or not isinstance(event.get("content"), str):
                            raise ValueError("Fragmento sin metadatos válidos.")
                        if sum(map(len, pieces)) + len(event["content"]) > 65536:
                            raise ValueError("La respuesta supera el límite del cliente.")
                        pieces.append(event["content"])
                        on_delta(event["content"])
                    elif kind == "done":
                        answer = "".join(pieces)
                        if metadata is None or not answer.strip() or type(event.get("characters")) is not int or event.get("characters") != len(answer) or event.get("chunks") != len(pieces):
                            raise ValueError("Flujo incompleto o métricas incoherentes.")
                        return {"answer": answer, **metadata, "charged_credits": validate_credit_receipt(event, authorization)}
                    else:
                        raise ValueError("Tipo de evento desconocido.")
                raise RuntimeError("La conexión terminó sin completar la respuesta.")

    async def run():
        if cancel.is_set():
            raise GenerationCancelled("Generación cancelada.")
        task = asyncio.create_task(consume())
        try:
            while not task.done():
                if cancel.is_set():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise GenerationCancelled("Generación cancelada.")
                await asyncio.wait({task}, timeout=0.05)
            if cancel.is_set():
                raise GenerationCancelled("Generación cancelada.")
            return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return asyncio.run(run())
