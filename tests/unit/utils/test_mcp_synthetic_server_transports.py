"""End-to-end transport tests for create_mcp_tool against scripts/mcp_synthetic_server.py."""

import asyncio
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from griptape.artifacts import ErrorArtifact, ImageArtifact, ListArtifact

from griptape_nodes_library.utils.mcp_compat_tool import Mcp2CompatibleTool, MCPToolInitError
from griptape_nodes_library.utils.mcp_utils import create_mcp_tool

SERVER_SCRIPT = Path(__file__).parents[3] / "scripts" / "mcp_synthetic_server.py"
EXPECTED_ACTIVITIES = ["add", "echo_text", "fail", "get_secret_code", "get_status", "red_pixel"]
SERVER_START_TIMEOUT_SECONDS = 20.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            msg = f"Synthetic MCP server exited early with code {process.returncode}"
            raise RuntimeError(msg)
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    msg = f"Synthetic MCP server did not start listening on port {port}"
    raise TimeoutError(msg)


def _start_server(*args: str) -> Generator[int, None, None]:
    port = _free_port()
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, str(SERVER_SCRIPT), "--port", str(port), *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(port, process)
        yield port
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture(scope="module")
def http_port() -> Generator[int, None, None]:
    yield from _start_server("--transport", "streamable-http")


@pytest.fixture(scope="module")
def sse_port() -> Generator[int, None, None]:
    yield from _start_server("--transport", "sse")


@pytest.fixture(scope="module")
def empty_http_port() -> Generator[int, None, None]:
    yield from _start_server("--transport", "streamable-http", "--empty")


def _http_config(port: int) -> dict[str, Any]:
    return {
        "transport": "streamable_http",
        "url": f"http://127.0.0.1:{port}/mcp",
        "headers": {},
        "timeout": 30.0,
        "sse_read_timeout": None,
        "terminate_on_close": True,
    }


def _sse_config(port: int) -> dict[str, Any]:
    return {"transport": "sse", "url": f"http://127.0.0.1:{port}/sse", "headers": {}, "timeout": 30.0}


def _stdio_config() -> dict[str, Any]:
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_SCRIPT), "--transport", "stdio"],
        "env": {},
        "cwd": None,
        "encoding": "utf-8",
        "encoding_error_handler": "strict",
    }


def _assert_tool_round_trip(tool: Mcp2CompatibleTool) -> None:
    assert sorted(tool.activity_name(a) for a in tool.activities()) == EXPECTED_ACTIVITIES
    assert tool.echo_text({"values": {"text": "hello"}}).to_text() == "hello"  # type: ignore[attr-defined]
    assert tool.add({"values": {"a": 2, "b": 3}}).to_text() == "5"  # type: ignore[attr-defined]
    assert tool.get_secret_code({"values": {}}).to_text() == "GTN-MCP-7319"  # type: ignore[attr-defined]

    image_result = tool.red_pixel({"values": {}})  # type: ignore[attr-defined]
    assert isinstance(image_result, ListArtifact)
    assert isinstance(image_result.value[0], ImageArtifact)
    assert image_result.value[0].format == "png"

    assert isinstance(tool.fail({"values": {"message": "boom"}}), ErrorArtifact)  # type: ignore[attr-defined]


class TestSyntheticServerTransports:
    def test_streamable_http(self, http_port: int) -> None:
        _assert_tool_round_trip(create_mcp_tool("synthetic", _http_config(http_port)))

    def test_streamable_http_inside_running_event_loop(self, http_port: int) -> None:
        async def _run() -> None:
            _assert_tool_round_trip(create_mcp_tool("synthetic", _http_config(http_port)))

        asyncio.run(_run())

    def test_sse(self, sse_port: int) -> None:
        _assert_tool_round_trip(create_mcp_tool("synthetic", _sse_config(sse_port)))

    def test_stdio(self) -> None:
        _assert_tool_round_trip(create_mcp_tool("synthetic", _stdio_config()))

    def test_server_with_no_tools_raises(self, empty_http_port: int) -> None:
        with pytest.raises(MCPToolInitError, match="did not expose any tools"):
            create_mcp_tool("synthetic", _http_config(empty_http_port))

    def test_unreachable_server_raises(self) -> None:
        with pytest.raises(MCPToolInitError, match="Failed to load tools from MCP server"):
            create_mcp_tool("synthetic", _http_config(_free_port()))
