"""Tests for the mcp 2.x compatible MCPTool and create_mcp_tool."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
from griptape.artifacts import ErrorArtifact, ImageArtifact, ListArtifact, TextArtifact
from mcp import types

from griptape_nodes_library.utils.mcp_compat_tool import (
    Mcp2CompatibleTool,
    MCPToolInitError,
    _create_streamable_http_session,
)
from griptape_nodes_library.utils.mcp_utils import create_mcp_tool

SERVER_CONFIG = {"transport": "streamable_http", "url": "http://localhost:8000/mcp", "headers": {}, "timeout": 30.0}

ECHO_TOOL = types.Tool(
    name="echo-text",
    description="Echo text back",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)


class FakeSession:
    def __init__(self, tools: list[types.Tool], call_result: Any = None) -> None:
        self.tools = tools
        self.call_result = call_result
        self.calls: list[tuple[str, dict]] = []

    async def initialize(self) -> None:
        pass

    async def list_tools(self) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.tools)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        return self.call_result


def _patch_session(session: FakeSession) -> Any:
    @asynccontextmanager
    async def _session(_self: Mcp2CompatibleTool) -> AsyncIterator[FakeSession]:
        yield session

    return patch.object(Mcp2CompatibleTool, "_get_session", _session)


def _patch_failing_session(exc: BaseException) -> Any:
    @asynccontextmanager
    async def _session(_self: Mcp2CompatibleTool) -> AsyncIterator[FakeSession]:
        raise exc
        yield  # pragma: no cover

    return patch.object(Mcp2CompatibleTool, "_get_session", _session)


class TestCreateMcpTool:
    def test_builds_activities_from_snake_case_input_schema(self) -> None:
        with _patch_session(FakeSession([ECHO_TOOL])):
            tool = create_mcp_tool("unreal-mcp", SERVER_CONFIG)

        assert isinstance(tool, Mcp2CompatibleTool)
        assert [tool.activity_name(a) for a in tool.activities()] == ["echo_text"]
        schema = tool.to_activity_json_schema(tool.activities()[0], "echo")
        assert "text" in schema["properties"]

    def test_initializes_inside_running_event_loop(self) -> None:
        async def _create() -> Mcp2CompatibleTool:
            return create_mcp_tool("unreal-mcp", SERVER_CONFIG)

        with _patch_session(FakeSession([ECHO_TOOL])):
            tool = asyncio.run(_create())

        assert len(tool.activities()) == 1

    def test_connection_failure_raises_instead_of_empty_tool(self) -> None:
        with (
            _patch_failing_session(ExceptionGroup("tg", [ConnectionError("refused")])),
            pytest.raises(MCPToolInitError, match="ConnectionError: refused"),
        ):
            create_mcp_tool("unreal-mcp", SERVER_CONFIG)

    def test_server_with_no_tools_raises(self) -> None:
        with _patch_session(FakeSession([])), pytest.raises(MCPToolInitError, match="did not expose any tools"):
            create_mcp_tool("unreal-mcp", SERVER_CONFIG)


class TestActivityExecution:
    def test_activity_calls_server_with_original_tool_name(self) -> None:
        session = FakeSession([ECHO_TOOL], types.CallToolResult(content=[types.TextContent(type="text", text="hello")]))
        with _patch_session(session):
            tool = create_mcp_tool("unreal-mcp", SERVER_CONFIG)
            result = tool.echo_text({"values": {"text": "hello"}})  # type: ignore[attr-defined]

        assert session.calls == [("echo-text", {"text": "hello"})]
        assert isinstance(result, ListArtifact)
        assert result.to_text() == "hello"


class TestConvertCallToolResult:
    @pytest.fixture
    def tool(self) -> Mcp2CompatibleTool:
        with _patch_session(FakeSession([ECHO_TOOL])):
            return create_mcp_tool("unreal-mcp", SERVER_CONFIG)

    def test_error_result_uses_text_content(self, tool: Mcp2CompatibleTool) -> None:
        result = tool._convert_call_tool_result_to_artifact(
            types.CallToolResult(content=[types.TextContent(type="text", text="boom")], is_error=True)
        )
        assert isinstance(result, ErrorArtifact)
        assert result.value == "boom"

    def test_error_result_falls_back_to_structured_content(self, tool: Mcp2CompatibleTool) -> None:
        result = tool._convert_call_tool_result_to_artifact(
            types.CallToolResult(content=[], structured_content={"code": 7}, is_error=True)
        )
        assert isinstance(result, ErrorArtifact)
        assert result.value == '{"code": 7}'

    def test_structured_content_used_when_no_content_blocks(self, tool: Mcp2CompatibleTool) -> None:
        result = tool._convert_call_tool_result_to_artifact(
            types.CallToolResult(content=[], structured_content={"ok": True})
        )
        assert isinstance(result, ListArtifact)
        assert isinstance(result.value[0], TextArtifact)
        assert result.value[0].value == '{"ok": true}'

    def test_image_format_strips_mime_prefix_only(self, tool: Mcp2CompatibleTool) -> None:
        result = tool._convert_call_tool_result_to_artifact(
            types.CallToolResult(content=[types.ImageContent(type="image", data="aGk=", mime_type="image/gif")])
        )
        assert isinstance(result, ListArtifact)
        image = result.value[0]
        assert isinstance(image, ImageArtifact)
        assert image.format == "gif"

    def test_non_call_tool_result_is_error(self, tool: Mcp2CompatibleTool) -> None:
        result = tool._convert_call_tool_result_to_artifact(object())
        assert isinstance(result, ErrorArtifact)
        assert "Unsupported MCP tool result type" in result.value


class TestStreamableHttpSession:
    def test_uses_mcp2_streamable_http_client_with_configured_http_client(self) -> None:
        captured: dict[str, Any] = {}

        @asynccontextmanager
        async def _fake_client(url: str, *, http_client: Any, terminate_on_close: bool) -> AsyncIterator[tuple]:
            captured.update(url=url, http_client=http_client, terminate_on_close=terminate_on_close)
            raise ConnectionError("stop")
            yield  # pragma: no cover

        async def _open() -> None:
            async with _create_streamable_http_session(
                {**SERVER_CONFIG, "headers": {"X-Test": "1"}, "terminate_on_close": False}
            ):
                pass  # pragma: no cover

        with (
            patch("griptape_nodes_library.utils.mcp_compat_tool.streamable_http_client", _fake_client),
            pytest.raises(ConnectionError),
        ):
            asyncio.run(_open())

        assert captured["url"] == "http://localhost:8000/mcp"
        assert captured["terminate_on_close"] is False
        assert captured["http_client"].headers["X-Test"] == "1"
        assert captured["http_client"].timeout.connect == 30.0  # noqa: PLR2004
