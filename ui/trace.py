"""Run the agent with streaming and render every step as it happens.

smolagents' `agent.run(stream=True)` yields PlanningStep / ActionStep objects as
each step finishes, ChatMessageStreamDelta tokens while the model is writing
(when stream_outputs is on), and a FinalAnswerStep last. Each step is turned
into a plain dict so the chat history can replay the trace after a rerun.
"""

import os
import re
import time

import plotly.io as pio
import streamlit as st
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep
from smolagents.models import ChatMessageStreamDelta

from tools import interactive_path, pop_charts

CHART_RE = re.compile(r"\[CHART:([^\]]+)\]")
_CODE_START = re.compile(r"<code>|```(?:python|py)?", re.IGNORECASE)


def _thought(model_output) -> str:
    text = model_output if isinstance(model_output, str) else str(model_output or "")
    text = _CODE_START.split(text, 1)[0]
    return re.sub(r"^\s*Thought:\s*", "", text).strip()


def _tokens(step) -> tuple[int, int]:
    usage = getattr(step, "token_usage", None)
    if usage is None:
        return 0, 0
    return usage.input_tokens or 0, usage.output_tokens or 0


def _duration(step) -> float:
    timing = getattr(step, "timing", None)
    return float(getattr(timing, "duration", None) or 0.0)


def step_record(step) -> dict:
    tin, tout = _tokens(step)
    if isinstance(step, PlanningStep):
        return {"kind": "plan", "plan": str(step.plan), "duration": _duration(step), "in": tin, "out": tout}
    return {
        "kind": "action",
        "n": step.step_number,
        "thought": _thought(step.model_output),
        "code": step.code_action or "",
        "observations": (step.observations or "").strip(),
        "error": str(step.error) if step.error else "",
        "duration": _duration(step),
        "in": tin,
        "out": tout,
    }


def render_step(rec: dict) -> None:
    if rec["kind"] == "plan":
        st.markdown("**Plan**")
        st.markdown(rec["plan"])
        st.caption(f"{rec['duration']:.1f}s")
        return
    st.markdown(f"**Step {rec['n']}**")
    if rec["thought"]:
        st.markdown(rec["thought"])
    if rec["code"]:
        st.code(rec["code"], language="python", wrap_lines=True)
    if rec["observations"]:
        with st.expander("Tool output", expanded=False):
            st.code(CHART_RE.sub("[chart]", rec["observations"]), language=None, wrap_lines=True)
    if rec["error"]:
        st.error(rec["error"])
    st.caption(f"{rec['duration']:.1f}s · {rec['in']:,} in / {rec['out']:,} out tokens")


def render_chart(path: str, key: str) -> None:
    """Interactive Plotly version if the tool saved one, otherwise the PNG."""
    ipath = interactive_path(path)
    if os.path.exists(ipath):
        try:
            st.plotly_chart(pio.read_json(ipath), width="stretch", key=key)
            return
        except Exception:
            pass
    if os.path.exists(path):
        st.image(path, width="stretch")


def render_message(msg: dict, idx: int) -> None:
    """Replay one stored chat message (user or assistant)."""
    with st.chat_message(msg["role"]):
        steps = msg.get("steps") or []
        if steps:
            label = f"Agent trace · {len(steps)} step{'s' if len(steps) != 1 else ''} · {msg.get('duration', 0):.1f}s"
            with st.status(label, state="error" if msg.get("failed") else "complete", expanded=False):
                for rec in steps:
                    render_step(rec)
        if msg.get("content"):
            st.markdown(msg["content"])
        for j, path in enumerate(msg.get("charts") or []):
            render_chart(path, key=f"msg{idx}-chart{j}")
        if msg.get("meta"):
            st.caption(msg["meta"])


def run_streaming(agent, prompt: str, reset: bool, live_tokens: bool) -> dict:
    """Run the agent, rendering each step live. Returns the assistant message dict
    (content, steps, charts, meta) and the run record under "run"."""
    steps: list[dict] = []
    answer, failed, error = "", False, ""
    started = time.time()
    status = st.status("Thinking…", expanded=True, type="step")
    with status:
        live = st.empty()
        buffer = ""
        try:
            for event in agent.run(prompt, stream=True, reset=reset):
                if isinstance(event, ChatMessageStreamDelta):
                    if live_tokens:
                        buffer += event.content or ""
                        live.markdown(buffer + " ▌")
                    continue
                if isinstance(event, (ActionStep, PlanningStep)):
                    live.empty()
                    buffer = ""
                    rec = step_record(event)
                    steps.append(rec)
                    with st.container():
                        render_step(rec)
                    label = "Planning…" if rec["kind"] == "plan" else f"Step {rec['n']} done, working…"
                    status.update(label=label)
                elif isinstance(event, FinalAnswerStep):
                    answer = str(event.output)
        except Exception as e:  # model/API failure: keep what we have, show the error
            failed, error = True, str(e)
            st.error(f"Run stopped: {e}")
    duration = time.time() - started
    status.update(
        label=f"Agent trace · {len(steps)} step{'s' if len(steps) != 1 else ''} · {duration:.1f}s",
        state="error" if failed else "complete",
        expanded=False,
    )

    # charts: registry first (reliable), then any markers the model copied through
    charts = pop_charts()
    for p in CHART_RE.findall(answer):
        p = p.strip()
        if p not in charts:
            charts.append(p)
    content = CHART_RE.sub("", answer).strip() or (f"Sorry, the run failed: {error}" if failed else "")
    tin = sum(s["in"] for s in steps)
    tout = sum(s["out"] for s in steps)
    meta = f"{len(steps)} steps · {duration:.1f}s · {tin:,} in / {tout:,} out tokens"
    return {
        "role": "assistant",
        "content": content,
        "steps": steps,
        "charts": charts,
        "duration": duration,
        "failed": failed,
        "meta": meta,
        "run": {
            "prompt": prompt,
            "status": "failed" if failed else "ok",
            "error": error,
            "steps": len(steps),
            "duration_s": round(duration, 2),
            "input_tokens": tin,
            "output_tokens": tout,
            "charts": len(charts),
            "answer": content,
            "trace": steps,
        },
    }
