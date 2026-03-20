"""Web-based trace viewer for Lakshya QAI agent runs.

Provides a collapsible, expandable web UI showing:
- Each agent invocation with timing and cost
- Every tool call with arguments (expandable)
- Tool results and errors (expandable)
- Full text responses
- Thinking blocks

Usage:
    python -m lakshya_qai.trace_viewer                    # view all sessions
    python -m lakshya_qai.trace_viewer --port 8050        # custom port
    python -m lakshya_qai.trace_viewer --session <id>     # view specific session

Then open http://localhost:8050 in your browser.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    list_sessions,
    get_session_messages,
    SDKSessionInfo,
    SessionMessage,
)

# ── HTML Template ────────────────────────────────────────────────────────

PAGE_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 10px; }
h2 { color: #79c0ff; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.session-card { background: #161b22; border: 1px solid #21262d; border-radius: 6px;
                padding: 16px; margin: 8px 0; cursor: pointer; }
.session-card:hover { border-color: #58a6ff; }
.session-meta { color: #8b949e; font-size: 0.85em; }
.message { margin: 8px 0; padding: 12px 16px; border-radius: 6px; border-left: 3px solid; }
.msg-user { background: #0d2137; border-color: #58a6ff; }
.msg-assistant { background: #1a1e24; border-color: #3fb950; }
.msg-result { background: #1c1917; border-color: #d29922; }
.msg-label { font-size: 0.75em; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
.msg-label-user { color: #58a6ff; }
.msg-label-assistant { color: #3fb950; }
.msg-label-result { color: #d29922; }
details { margin: 4px 0; }
summary { cursor: pointer; padding: 6px 10px; background: #21262d; border-radius: 4px;
          font-size: 0.9em; user-select: none; }
summary:hover { background: #30363d; }
.tool-call { background: #1a1500; border: 1px solid #3d2e00; border-radius: 4px;
             padding: 10px; margin: 6px 0; }
.tool-name { color: #d29922; font-weight: bold; font-size: 0.95em; }
.tool-result { background: #0a1628; border: 1px solid #1a3050; border-radius: 4px;
               padding: 10px; margin: 6px 0; }
.tool-error { background: #2d0a0a; border: 1px solid #5a1a1a; }
.text-block { white-space: pre-wrap; font-size: 0.9em; line-height: 1.5; }
.thinking { color: #8b949e; font-style: italic; }
pre { background: #161b22; padding: 10px; border-radius: 4px; overflow-x: auto;
      font-size: 0.85em; border: 1px solid #21262d; }
code { font-family: 'Fira Code', 'Cascadia Code', monospace; }
.cost { color: #3fb950; font-weight: bold; }
.error-badge { background: #da3633; color: white; padding: 2px 6px; border-radius: 3px;
               font-size: 0.75em; }
.stats-bar { display: flex; gap: 20px; background: #161b22; padding: 10px 16px;
             border-radius: 6px; margin: 10px 0; font-size: 0.9em; }
.stat { display: flex; flex-direction: column; }
.stat-label { color: #8b949e; font-size: 0.75em; }
.stat-value { color: #c9d1d9; font-weight: bold; }
.back-link { display: inline-block; margin-bottom: 16px; padding: 6px 12px;
             background: #21262d; border-radius: 4px; }
"""


def _escape(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text)) if text else ""


def _format_json(obj: dict | list | str, max_length: int = 2000) -> str:
    """Pretty-format JSON for display."""
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"
    return _escape(text)


def _render_content_block(block: dict) -> str:
    """Render a single content block to HTML."""
    btype = block.get("type", "")

    if btype == "text":
        text = block.get("text", "")
        return f'<div class="text-block">{_escape(text)}</div>'

    elif btype == "thinking":
        thinking = block.get("thinking", "")
        return (
            f'<details><summary>Thinking ({len(thinking)} chars)</summary>'
            f'<div class="thinking text-block">{_escape(thinking)}</div></details>'
        )

    elif btype == "tool_use":
        name = block.get("name", "?")
        input_data = block.get("input", {})
        tool_id = block.get("id", "")
        return (
            f'<div class="tool-call">'
            f'<span class="tool-name">TOOL CALL: {_escape(name)}</span>'
            f' <span style="color:#8b949e;font-size:0.8em">({_escape(tool_id)})</span>'
            f'<details><summary>Arguments</summary>'
            f'<pre><code>{_format_json(input_data)}</code></pre></details>'
            f'</div>'
        )

    elif btype == "tool_result":
        tool_id = block.get("tool_use_id", "")
        content = block.get("content", "")
        is_error = block.get("is_error", False)
        error_class = " tool-error" if is_error else ""
        error_badge = ' <span class="error-badge">ERROR</span>' if is_error else ""
        if isinstance(content, list):
            # MCP-style content blocks
            rendered = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    rendered += _escape(item.get("text", ""))
                else:
                    rendered += _format_json(item)
        else:
            rendered = _format_json(content) if content else "(empty)"
        return (
            f'<div class="tool-result{error_class}">'
            f'<span style="color:#58a6ff;font-size:0.85em">RESULT for {_escape(tool_id)}</span>'
            f'{error_badge}'
            f'<details><summary>Response ({len(str(content))} chars)</summary>'
            f'<pre><code>{rendered}</code></pre></details>'
            f'</div>'
        )

    return f'<pre>{_format_json(block)}</pre>'


def _render_message(msg: SessionMessage) -> str:
    """Render a single session message to HTML."""
    mtype = msg.type
    css_class = f"msg-{mtype}"
    label_class = f"msg-label-{mtype}"

    label = mtype.upper()
    if mtype == "assistant" and msg.parent_tool_use_id:
        label += f" (sub-agent)"

    # msg.message can be a dict or have 'content' key
    message_data = msg.message
    if isinstance(message_data, dict):
        content_blocks = message_data.get("content", [])
        model = message_data.get("model", "")
    else:
        content_blocks = []
        model = ""

    blocks_html = ""
    for block in content_blocks:
        if isinstance(block, dict):
            blocks_html += _render_content_block(block)

    model_badge = f' <span style="color:#8b949e;font-size:0.8em">[{_escape(model)}]</span>' if model else ""

    return (
        f'<div class="message {css_class}">'
        f'<div class="msg-label {label_class}">{label}{model_badge}</div>'
        f'{blocks_html}'
        f'</div>'
    )


def render_session_list(sessions: list[SDKSessionInfo]) -> str:
    """Render the session list page."""
    cards = ""
    for s in sessions:
        ts = datetime.fromtimestamp(s.last_modified / 1000).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = s.file_size / 1024
        prompt_preview = _escape(s.first_prompt[:120] + "..." if s.first_prompt and len(s.first_prompt) > 120 else (s.first_prompt or ""))
        cards += (
            f'<a href="/session/{s.session_id}">'
            f'<div class="session-card">'
            f'<strong>{_escape(s.summary or s.session_id)}</strong><br>'
            f'<div class="session-meta">'
            f'{ts} | {size_kb:.0f} KB | Branch: {_escape(s.git_branch or "?")}'
            f'</div>'
            f'<div style="margin-top:6px;color:#8b949e;font-size:0.85em">{prompt_preview}</div>'
            f'</div></a>'
        )

    return f"""<!DOCTYPE html>
<html><head><title>QAI Trace Viewer</title><style>{PAGE_CSS}</style></head>
<body>
<h1>Lakshya QAI — Agent Trace Viewer</h1>
<p style="color:#8b949e">{len(sessions)} sessions found. Click to inspect.</p>
{cards}
</body></html>"""


def render_session_detail(session_id: str, messages: list[SessionMessage]) -> str:
    """Render a single session's full trace."""
    msgs_html = ""
    for msg in messages:
        msgs_html += _render_message(msg)

    return f"""<!DOCTYPE html>
<html><head><title>Session {_escape(session_id[:12])}</title><style>{PAGE_CSS}</style></head>
<body>
<a href="/" class="back-link">Back to sessions</a>
<h1>Session: {_escape(session_id[:20])}...</h1>
<div class="stats-bar">
    <div class="stat"><span class="stat-label">Messages</span><span class="stat-value">{len(messages)}</span></div>
</div>
{msgs_html}
</body></html>"""


# ── FastAPI Server ───────────────────────────────────────────────────────

def create_app(project_dir: str | None = None):
    """Create the FastAPI trace viewer app."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="QAI Trace Viewer")

    @app.get("/", response_class=HTMLResponse)
    async def session_list():
        sessions = list_sessions(directory=project_dir, limit=50)
        sessions.sort(key=lambda s: s.last_modified, reverse=True)
        return render_session_list(sessions)

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    async def session_detail(session_id: str):
        messages = get_session_messages(session_id, directory=project_dir)
        return render_session_detail(session_id, messages)

    return app


def main():
    parser = argparse.ArgumentParser(description="QAI Agent Trace Viewer")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dir", default=None, help="Project directory to scan for sessions")
    args = parser.parse_args()

    import uvicorn
    app = create_app(project_dir=args.dir)
    print(f"\nTrace viewer at http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
