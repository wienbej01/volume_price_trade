#!/usr/bin/env python3
import os, sys, json, pathlib, subprocess, shutil, traceback
from datetime import datetime
from typing import Any, Dict, List

# pip install openai>=1.40 pyyaml
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root = ~/volume_price_trade
SAFE_ROOT = ROOT.resolve()

SYSTEM_POLICY = f"""
You are the **Orchestrator** for the local repository at: {SAFE_ROOT}

You MUST:
- Work ONLY inside this repo root. Refuse any path that escapes it.
- Prefer small, reversible changes; keep YAML/JSON valid.
- After completing each step, follow the user's prompt acceptance rules.
- All heavy operations must be done via tools provided (write_file, mkdirs, append_file, update_yaml, run).
- Use 'run' to execute shell commands (tests, formatters, git, python scripts).
- When you claim a command was run, you must actually call the 'run' tool.

Constraints:
- Do not access the network.
- Do not exfiltrate secrets.
- Use UTC timestamps in logs/state when asked.

General pattern per step:
1) Plan the edits and commands.
2) Call tools in the required order (create dirs/files, run checks, commit).
3) Summarize what changed and what to do next.
"""

TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "mkdirs",
      "description": "Create directories (parents ok) under repo root.",
      "parameters": {
        "type": "object",
        "properties": {
          "paths": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["paths"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Create/overwrite a text file under repo root.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "content": {"type": "string"}
        },
        "required": ["path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "append_file",
      "description": "Append text to a file under repo root; creates file if missing.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "content": {"type": "string"}
        },
        "required": ["path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "update_yaml",
      "description": "Update a YAML file by replacing it entirely with provided text (model must produce valid YAML).",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "new_yaml": {"type": "string"}
        },
        "required": ["path", "new_yaml"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "run",
      "description": "Run a shell command from repo root. Capture stdout/stderr/exit code.",
      "parameters": {
        "type": "object",
        "properties": {
          "cmd": {"type": "string"}
        },
        "required": ["cmd"]
      }
    }
  }
]

def _safe_path(p: str) -> pathlib.Path:
    rp = (SAFE_ROOT / p).resolve()
    if not str(rp).startswith(str(SAFE_ROOT)):
        raise ValueError(f"Unsafe path outside repo: {p}")
    return rp

def mkdirs(paths: List[str]) -> Dict[str, Any]:
    created = []
    for p in paths:
        rp = _safe_path(p)
        rp.mkdir(parents=True, exist_ok=True)
        created.append(str(rp.relative_to(SAFE_ROOT)))
    return {"created": created}

def write_file(path: str, content: str) -> Dict[str, Any]:
    rp = _safe_path(path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(content, encoding="utf-8")
    return {"wrote": str(rp.relative_to(SAFE_ROOT)), "bytes": len(content.encode("utf-8"))}

def append_file(path: str, content: str) -> Dict[str, Any]:
    rp = _safe_path(path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with rp.open("a", encoding="utf-8") as f:
        f.write(content)
    return {"appended": str(rp.relative_to(SAFE_ROOT)), "bytes": len(content.encode("utf-8"))}

def update_yaml(path: str, new_yaml: str) -> Dict[str, Any]:
    # Simple full-replacement; validation is up to the model/tests.
    return write_file(path, new_yaml)

def run(cmd: str) -> Dict[str, Any]:
    proc = subprocess.run(cmd, shell=True, cwd=str(SAFE_ROOT),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"cmd": cmd, "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:], "stderr": proc.stderr[-8000:]}

TOOL_IMPL = {
    "mkdirs": mkdirs,
    "write_file": write_file,
    "append_file": append_file,
    "update_yaml": update_yaml,
    "run": run,
}

def call_model(client: OpenAI, messages: List[Dict[str, str]]):
    return client.chat.completions.create(
        model="z-ai/glm-4.5",
        messages=messages,
        tools=TOOLS,
        reasoning={"enabled": True},
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: scripts/glm_orchestrator.py codex/prompts/00_persona.md", file=sys.stderr)
        sys.exit(1)

    prompt_file = sys.argv[1]
    pf = _safe_path(prompt_file)
    user_prompt = pf.read_text(encoding="utf-8")

    # Env
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        print("OPENROUTER_API_KEY not set. source ./env.sh first.", file=sys.stderr)
        sys.exit(2)

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_POLICY},
        {"role": "user", "content": user_prompt},
    ]

    # Tool-call loop
    for _ in range(40):  # generous cap to avoid accidental infinite loops
        resp = call_model(client, messages)
        choice = resp.choices[0]

        # Emit assistant content if any
        msg = choice.message
        if msg.content:
            print("\n[assistant]:\n" + msg.content.strip() + "\n")

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            # No more tools -> we are done
            break

        # Execute tools sequentially
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            try:
                result = TOOL_IMPL[name](**args)
            except Exception as e:
                result = {"error": str(e), "trace": traceback.format_exc(-1)}

            # Send tool result back
            messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})  # keep context
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # Also append the assistant message to history so model sees what it wrote
        if msg.content:
            messages.append({"role": "assistant", "content": msg.content})

    # done

if __name__ == "__main__":
    main()
