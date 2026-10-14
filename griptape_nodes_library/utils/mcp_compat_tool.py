"""MCPTool subclass that works with the mcp 2.x SDK.

griptape 1.13's `MCPTool` was written against mcp 1.x, but griptape-nodes-engine pins
`mcp>=2.0,<3`. mcp 2.x broke the framework tool in two places:

- `mcp.client.streamable_http.streamablehttp_client` was replaced by `streamable_http_client`,
  which takes a pre-built `httpx2.AsyncClient` instead of headers/timeouts and yields a
  2-tuple of streams instead of a 3-tuple.
- Pydantic models switched to snake_case attributes (`Tool.input_schema`,
  `CallToolResult.is_error`/`structured_content`, `*Content.mime_type`).

The stock tool also initializes its activities on a background thread whose exceptions are
silently dropped, so any of the above left a tool with zero activities. Bedrock then rejected
the agent's request with "Invalid length for parameter toolConfig.tools, value: 0".

Remove this module once griptape ships an `MCPTool` that supports mcp 2.x.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx2
from attrs import define
from griptape.artifacts import (
    AudioArtifact,
    BaseArtifact,
    BlobArtifact,
    ErrorArtifact,
    ImageArtifact,
    ListArtifact,
    TextArtifact,
)
from griptape.tools import BaseTool, MCPTool
from griptape.tools.mcp.sessions import create_session
from griptape.utils.decorators import activity
from json_schema_to_pydantic import create_model
from mcp import ClientSession, types
from mcp.client.streamable_http import (
    create_mcp_http_client,  # pyright: ignore[reportPrivateImportUsage]
    streamable_http_client,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import _AsyncGeneratorContextManager

DEFAULT_STREAMABLE_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT_SECONDS = 300.0


class MCPToolInitError(RuntimeError):
    """Raised when an MCP tool cannot connect to its server or list the server's tools."""


def _to_seconds(value: float | timedelta | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)


def _flatten_exception_message(exc: BaseException) -> str:
    """Join the messages of every leaf exception, unwrapping anyio's exception groups."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_flatten_exception_message(e) for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def _create_streamable_http_session(connection: dict[str, Any]) -> AsyncIterator[ClientSession]:
    timeout = _to_seconds(connection.get("timeout"), DEFAULT_STREAMABLE_HTTP_TIMEOUT_SECONDS)
    sse_read_timeout = _to_seconds(connection.get("sse_read_timeout"), DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT_SECONDS)
    http_client = create_mcp_http_client(
        headers=connection.get("headers") or None,
        timeout=httpx2.Timeout(timeout, read=sse_read_timeout),
    )
    async with (
        http_client,
        streamable_http_client(
            connection["url"],
            http_client=http_client,
            terminate_on_close=connection.get("terminate_on_close", True),
        ) as (read, write),
        ClientSession(read, write, **(connection.get("session_kwargs") or {})) as session,
    ):
        yield session


@define
class Mcp2CompatibleTool(MCPTool):
    """`MCPTool` that speaks the mcp 2.x client API and surfaces initialization failures."""

    def __attrs_post_init__(self) -> None:
        # Skip MCPTool.__attrs_post_init__: it swallows init errors raised on its worker thread.
        BaseTool.__attrs_post_init__(self)
        errors: list[BaseException] = []

        def _init() -> None:
            try:
                asyncio.run(self._init_activities())
            except BaseException as e:
                errors.append(e)

        # Always use a dedicated thread so this works whether or not an event loop is running.
        thread = threading.Thread(target=_init)
        thread.start()
        thread.join()
        if errors:
            msg = f"Failed to load tools from MCP server: {_flatten_exception_message(errors[0])}"
            raise MCPToolInitError(msg) from errors[0]

    def _get_session(self) -> _AsyncGeneratorContextManager[ClientSession, None]:
        connection = dict(self.connection)
        if connection.get("transport") == "streamable_http":
            if "url" not in connection:
                msg = "'url' parameter is required for Streamable HTTP connection"
                raise ValueError(msg)
            return _create_streamable_http_session(connection)
        # stdio and sse client signatures are unchanged in mcp 2.x.
        return create_session(self.connection)

    def _create_activity_handler(self, tool: types.Tool) -> Callable:
        activity_name = self._sanitize_activity_name(tool.name)

        @activity(
            config={
                "name": activity_name,
                "description": tool.description or tool.title or tool.name,
                "schema": create_model(tool.input_schema, allow_undefined_array_items=True, allow_undefined_type=True),
            }
        )
        def activity_handler(self: Mcp2CompatibleTool, values: dict) -> Any:
            result: list[BaseArtifact] = []

            def _run() -> None:
                result.append(asyncio.run(self._run_activity(tool.name, values)))

            thread = threading.Thread(target=_run)
            thread.start()
            thread.join()
            return result[0] if result else ErrorArtifact(f"MCP tool '{tool.name}' returned no result.")

        return activity_handler

    def _convert_error_result_to_artifact(self, call_tool_result: types.CallToolResult) -> ErrorArtifact:
        error_text = next(
            (content.text for content in call_tool_result.content if isinstance(content, types.TextContent)), None
        )
        if error_text is None and call_tool_result.structured_content is not None:
            error_text = json.dumps(call_tool_result.structured_content)
        return ErrorArtifact(error_text or "An unknown error occurred.")

    def _convert_call_tool_result_to_artifact(self, call_tool_result: Any) -> ListArtifact | ErrorArtifact:
        if not isinstance(call_tool_result, types.CallToolResult):
            # mcp 2.x can also return InputRequiredResult (elicitation), which agents can't answer.
            return ErrorArtifact(f"Unsupported MCP tool result type: {type(call_tool_result).__name__}")
        if call_tool_result.is_error:
            return self._convert_error_result_to_artifact(call_tool_result)

        response_artifacts: list[BaseArtifact] = []
        for content in call_tool_result.content:
            if isinstance(content, types.TextContent):
                response_artifacts.append(TextArtifact(content.text))
            elif isinstance(content, types.ImageContent):
                response_artifacts.append(
                    ImageArtifact(
                        value=content.data, format=content.mime_type.removeprefix("image/"), width=0, height=0
                    )
                )
            elif isinstance(content, types.AudioContent):
                response_artifacts.append(
                    AudioArtifact(value=content.data, format=content.mime_type.removeprefix("audio/"))
                )
            elif isinstance(content, types.EmbeddedResource):
                if isinstance(content.resource, types.TextResourceContents):
                    response_artifacts.append(TextArtifact(content.resource.text))
                elif isinstance(content.resource, types.BlobResourceContents):
                    response_artifacts.append(BlobArtifact(value=content.resource.blob))

        if not response_artifacts and call_tool_result.structured_content is not None:
            response_artifacts.append(TextArtifact(json.dumps(call_tool_result.structured_content)))

        return ListArtifact(response_artifacts)
