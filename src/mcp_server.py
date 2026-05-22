"""MCP server for Verbatim.

Exposes the same four tools as the agent pipeline via the Model Context
Protocol, so Claude Desktop (or any MCP client) can connect directly to
your interview corpus. With this running, you can open Claude on your
laptop and ask "what are users saying about pricing?" and Claude will
call into this server, search your transcripts, and answer.

Why this matters for the project:
  Building the same retrieval logic twice (once for the in-process agent,
  once for MCP) would be a bad sign. Instead, we share the implementations
  — the MCP tools call the same underlying functions the agent calls.
  This is the right shape: one set of tools, multiple front-ends.

To use:
  python -m src.mcp_server

  Then in Claude Desktop's config (~/Library/Application Support/Claude/
  claude_desktop_config.json), add:

  {
    "mcpServers": {
      "verbatim": {
        "command": "python",
        "args": ["-m", "src.mcp_server"],
        "cwd": "/absolute/path/to/verbatim"
      }
    }
  }

  Restart Claude Desktop and you'll see the tools appear.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from .agent import AgentPipeline


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def main() -> None:
    # Late imports so the file is at least loadable without the mcp package.
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise SystemExit(
            "The `mcp` package isn't installed. Run: pip install mcp"
        )

    # The agent has all four tools implemented; we just expose them.
    agent = AgentPipeline()
    server = Server("verbatim")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_interviews",
                description=(
                    "Semantic search over customer-research interview turns. "
                    "Best for thematic questions and finding evidence across "
                    "the corpus. Returns top-K most relevant speaker turns."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_full_transcript",
                description=(
                    "Load the full text of one interview by interview_id "
                    "(e.g. '05_diana'). Use for questions about a specific "
                    "named person."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"interview_id": {"type": "string"}},
                    "required": ["interview_id"],
                },
            ),
            Tool(
                name="find_quotes",
                description=(
                    "Literal substring search for exact phrases or quotes. "
                    "Case-insensitive."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["phrase"],
                },
            ),
            Tool(
                name="list_interviews",
                description=(
                    "Directory of every interview: id, participant, role, "
                    "and recruitment basis."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = agent._dispatch_tool(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
