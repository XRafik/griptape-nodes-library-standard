"""Report a provider's HTTP error as a NodeError.

The message is the provider's own explanation. Its error body, status code, error code, and
request ID are attached, so the editor's error panel shows them on their own lines.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.utils.node_error_utils import (
    error_fields,
    error_response,
    parse_error_body,
    provider_error_fields,
    provider_error_message,
    request_id_from_headers,
)

# How far down `__cause__` to look for the provider SDK's error under pydantic-ai's.
_MAX_CHAIN = 4


def provider_node_error(exc: BaseException) -> NodeError | None:
    """The NodeError for a provider's HTTP error, or None when `exc` is not one."""
    if isinstance(exc, NodeError):
        return None
    chain = _cause_chain(exc)
    status_code = next((code for e in chain if (code := _status_code(e)) is not None), None)
    if status_code is None:
        return None
    body = next((b for e in chain if (b := _body(e)) is not None), None)
    body = _embedded_body(chain, body) or body
    bedrock = _bedrock_error(body)
    message = provider_error_message(body) or bedrock.get("Message") or _fallback_message(chain)
    fields = error_fields(status_code=status_code)
    fields.update(provider_error_fields(body))
    if "error_code" not in fields and isinstance(bedrock.get("Code"), str):
        fields["error_code"] = bedrock["Code"]
    request_id = _request_id(chain, body)
    if request_id and "request_id" not in fields:
        fields["request_id"] = request_id
    return NodeError(message, fields=fields, response=error_response(body))


def raise_provider_error(exc: BaseException) -> None:
    """Raise the NodeError for `exc`, if it is a provider's HTTP error."""
    error = provider_node_error(exc)
    if error is not None:
        raise error from exc


def _cause_chain(exc: BaseException) -> list[BaseException]:
    chain = [exc]
    while len(chain) < _MAX_CHAIN and (cause := chain[-1].__cause__) is not None:
        chain.append(cause)
    return chain


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if not isinstance(code, int):
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) and code >= 400 else None  # noqa: PLR2004


def _body(exc: BaseException) -> dict[str, Any] | None:
    """The JSON error body from the exception's HTTP response, or the body an SDK parsed."""
    read_json = getattr(getattr(exc, "response", None), "json", None)
    if callable(read_json):
        try:
            body = read_json()
        # A body that isn't JSON, or a streamed response that was never read.
        except (ValueError, RuntimeError):
            body = None
        if isinstance(body, dict) and body:
            return body
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body:
        return body
    # botocore's ClientError keeps the parsed body as `.response`.
    response = getattr(exc, "response", None)
    return response if isinstance(response, dict) and response else None


def _embedded_body(chain: list[BaseException], body: dict[str, Any] | None) -> dict[str, Any] | None:
    """The provider body Griptape Cloud passed on as text, wherever it ended up.

    It can be a string under the body's "error", "detail", or "message" key, the raw response
    text, or only an exception's message.
    """
    candidates: list[Any] = [body.get(key) for key in ("error", "detail", "message")] if body else []
    for exc in chain:
        candidates.extend([_response_text(exc), str(exc)])
    for candidate in candidates:
        if isinstance(candidate, str):
            embedded = parse_error_body(candidate)
            if embedded is not None:
                return embedded
    return None


def _bedrock_error(body: dict[str, Any] | None) -> dict[str, Any]:
    error = body.get("Error") if body else None
    return error if isinstance(error, dict) else {}


def _request_id(chain: list[BaseException], body: dict[str, Any] | None) -> str | None:
    for exc in chain:
        request_id = getattr(exc, "request_id", None)
        if isinstance(request_id, str) and request_id:
            return request_id
        headers = getattr(exc, "headers", None) or getattr(getattr(exc, "response", None), "headers", None)
        request_id = request_id_from_headers(headers) if hasattr(headers, "get") else None
        if request_id:
            return request_id
    metadata = body.get("ResponseMetadata") if body else None
    request_id = metadata.get("RequestId") if isinstance(metadata, dict) else None
    return request_id if isinstance(request_id, str) and request_id else None


def _response_text(exc: BaseException) -> str | None:
    try:
        text = getattr(getattr(exc, "response", None), "text", None)
    # A streamed response that was never read.
    except RuntimeError:
        return None
    return text if isinstance(text, str) else None


def _fallback_message(chain: list[BaseException]) -> str:
    text = next((t.strip() for e in chain if (t := _response_text(e)) and t.strip()), None)
    return f"The model request failed: {text or chain[-1]}"
