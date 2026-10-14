"""Synthetic MCP server for exercising the library's MCP nodes without a real backend.

Exposes a small, deterministic tool set covering the shapes Mcp2CompatibleTool has to handle:
hyphenated tool names, text/structured/image results, and tool errors.

Usage:
    uv run python scripts/mcp_synthetic_server.py                                  # streamable HTTP on 127.0.0.1:18765/mcp
    uv run python scripts/mcp_synthetic_server.py --transport sse --port 18766      # SSE on 127.0.0.1:18766/sse
    uv run python scripts/mcp_synthetic_server.py --transport stdio                # stdio (spawned by the client)
    uv run python scripts/mcp_synthetic_server.py --empty                          # server that exposes no tools
"""

import argparse
import base64

from mcp.server.mcpserver import Image, MCPServer

# 1x1 red PNG.
RED_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)

SECRET_CODE = "GTN-MCP-7319"


def build_server(*, empty: bool) -> MCPServer:
    server = MCPServer(name="gtn-synthetic-mcp", instructions="Synthetic MCP server for Griptape Nodes testing.")
    if empty:
        return server

    @server.tool(name="echo-text", description="Echo the given text back unchanged.")
    def echo_text(text: str) -> str:
        return text

    @server.tool(description="Add two integers and return the sum.")
    def add(a: int, b: int) -> int:
        return a + b

    @server.tool(description="Return the secret verification code. Call this when asked for the secret code.")
    def get_secret_code() -> str:
        return SECRET_CODE

    @server.tool(description="Return a small structured status object.")
    def get_status() -> dict:
        return {"ok": True, "server": "gtn-synthetic-mcp", "tool_count": 6}

    @server.tool(description="Return a 1x1 red PNG image.")
    def red_pixel() -> Image:
        return Image(data=RED_PIXEL_PNG, format="png")

    @server.tool(description="Always fails with the given message. Use to test tool error handling.")
    def fail(message: str = "synthetic failure") -> str:
        raise RuntimeError(message)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["streamable-http", "sse", "stdio"], default="streamable-http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--empty", action="store_true", help="Expose no tools.")
    args = parser.parse_args()

    server = build_server(empty=args.empty)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(args.transport, host=args.host, port=args.port)  # type: ignore[call-overload]


if __name__ == "__main__":
    main()
