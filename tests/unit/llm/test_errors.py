from __future__ import annotations

import json
from typing import Any

import httpx
import httpx2
import openai
import pytest
from griptape_nodes.exe_types.core_types import NodeError
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from griptape_nodes_library.llm import models
from griptape_nodes_library.llm.errors import provider_node_error
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.utils.cloud_credential_utils import MissingCredentialError
from griptape_nodes_library.utils.node_error_utils import MissingSecretError


def _http_error(
    status: int, *, json_body: dict[str, Any] | None = None, text: str = "", request_id: str | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    headers = {"x-request-id": request_id} if request_id else {}
    content = json.dumps(json_body).encode() if json_body is not None else text.encode()
    response = httpx.Response(status, content=content, headers=headers, request=request)
    return httpx.HTTPStatusError(f"{status} Client Error", request=request, response=response)


def _pydantic_ai_error(status: int, body: dict[str, Any], *, request_id: str = "req_123") -> ModelHTTPError:
    """Shaped like pydantic-ai's error over an OpenAI SDK error: the SDK keeps the inner body."""
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(status, json=body, headers={"x-request-id": request_id}, request=request)
    sdk = openai.APIStatusError(f"Error code: {status} - {body!r}", response=response, body=body.get("error"))
    try:
        raise ModelHTTPError(status, "gpt-x", body=body.get("error"), headers=dict(response.headers)) from sdk
    except ModelHTTPError as exc:
        return exc


def test_provider_message_becomes_the_message_and_body_is_attached() -> None:
    body = {"error": {"message": "Invalid model 'gpt-9'.", "type": "invalid_request_error", "param": "model"}}

    error = provider_node_error(_http_error(400, json_body=body, request_id="req_abc"))

    assert error is not None
    assert str(error) == "Invalid model 'gpt-9'."
    assert error.response == body
    assert error.fields == {"status_code": 400, "parameter": "model", "request_id": "req_abc"}


def test_pydantic_ai_error_reads_the_sdk_error_under_it() -> None:
    body = {"error": {"message": "Prompt is too long.", "code": "context_length_exceeded"}}

    error = provider_node_error(_pydantic_ai_error(400, body))

    assert error is not None
    assert str(error) == "Prompt is too long."
    assert error.response == body
    assert error.fields == {"status_code": 400, "error_code": "context_length_exceeded", "request_id": "req_123"}


def test_bedrock_client_error_body_is_used() -> None:
    body = {
        "Error": {"Code": "AccessDeniedException", "Message": "You don't have access to the model."},
        "ResponseMetadata": {"RequestId": "aws-req-1", "HTTPStatusCode": 403},
    }

    error = provider_node_error(ModelHTTPError(403, "claude", body=body))

    assert error is not None
    assert str(error) == "You don't have access to the model."
    assert error.fields == {"status_code": 403, "error_code": "AccessDeniedException", "request_id": "aws-req-1"}


def test_non_json_body_falls_back_to_its_text() -> None:
    error = provider_node_error(_http_error(502, text="Bad Gateway"))

    assert error is not None
    assert str(error) == "The model request failed: Bad Gateway"
    assert error.response is None
    assert error.fields == {"status_code": 502}


_OPENAI_BODY = {
    "error": {
        "message": "Unsupported value: 'temperature' does not support 0.1 with this model.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(json.dumps({"error": f"Error code: 400 - {_OPENAI_BODY!r}"}), id="json-error-string"),
        pytest.param(json.dumps({"detail": f"Error code: 400 - {_OPENAI_BODY!r}"}), id="json-detail-string"),
        pytest.param(f"Error code: 400 - {_OPENAI_BODY!r}", id="plain-text"),
        pytest.param(json.dumps({"error": f'"Error code: 400 - {_OPENAI_BODY!r}"'}), id="quoted"),
    ],
)
def test_provider_body_griptape_cloud_passed_on_as_text_is_unwrapped(text: str) -> None:
    error = provider_node_error(_http_error(400, text=text))

    assert error is not None
    assert str(error) == "Unsupported value: 'temperature' does not support 0.1 with this model."
    assert error.response == _OPENAI_BODY
    assert error.fields == {"status_code": 400, "error_code": "unsupported_value", "parameter": "temperature"}


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(NodeError("Already worded."), id="node-error"),
        pytest.param(UnexpectedModelBehavior("Exceeded maximum output retries"), id="not-http"),
        pytest.param(ValueError("boom"), id="plain"),
    ],
)
def test_errors_that_are_not_provider_http_errors_are_left_alone(exc: Exception) -> None:
    assert provider_node_error(exc) is None


def test_missing_cloud_credential_links_to_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "resolve_cloud_api_key", lambda: "")

    with pytest.raises(MissingCredentialError, match="Griptape Cloud") as caught:
        models.build_model(ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1"))

    assert [link.url for link in caught.value.links] == ["#settings-secrets?filter=GT_CLOUD_API_KEY"]


def test_missing_provider_key_links_to_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "_secret", lambda _name: None)

    with pytest.raises(MissingSecretError, match="ANTHROPIC_API_KEY is not set") as caught:
        models.build_model(ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude"))

    assert [link.url for link in caught.value.links] == ["#settings-secrets?filter=ANTHROPIC_API_KEY"]
