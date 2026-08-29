"""
DataLens AI — Agent Orchestrator

Implements the investigation loop from MASTER PROMPT #16:

    Question -> Intent -> Plan -> Tool Call -> Evidence -> Evaluate
    -> (more evidence needed? -> another tool) -> Synthesize -> Answer

Model-agnostic: supports Anthropic tool-use, OpenAI function-calling, or a
deterministic no-LLM fallback (MASTER PROMPT #37) so the app never fully
breaks if the LLM API is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import config
from src.agent.evidence import Evidence, EvidenceChain
from src.agent.planner import build_initial_plan
from src.agent.prompts import SYSTEM_PROMPT, INVESTIGATION_STARTER_TEMPLATE, SYNTHESIS_TEMPLATE
from src.agent.tools import TOOL_REGISTRY, get_tool_schemas, get_openai_tool_schemas, call_tool

logger = logging.getLogger(__name__)


@dataclass
class InvestigationResult:
    question: str
    intent: str
    trace: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    answer: str = ""
    charts_needed: list[str] = field(default_factory=list)  # tool names whose results are chart-worthy
    mode: str = "llm"  # "llm" or "fallback"
    error: Optional[str] = None


def _evidence_from_tool_result(tool_name: str, args: dict, result: dict) -> Evidence:
    return Evidence(
        metric=tool_name,
        value=result.get("result"),
        source_tool=tool_name,
        raw_result={"args": args, **result},
    )


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------
def _run_anthropic_investigation(question: str, plan: dict, chain: EvidenceChain) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    tools = get_tool_schemas()

    messages = [{
        "role": "user",
        "content": INVESTIGATION_STARTER_TEMPLATE.format(
            question=question, intent=plan["intent"], suggested_tools=plan["suggested_tools"],
        ),
    }]

    for step in range(config.MAX_INVESTIGATION_STEPS):
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            # Model produced a direct text answer without (further) tool calls.
            text = "".join(b.text for b in response.content if b.type == "text")
            if text.strip():
                return text
            break

        tool_results_content = []
        for call in tool_calls:
            chain.add_trace(f"Called `{call.name}` with {json.dumps(call.input)}")
            result = call_tool(call.name, call.input)
            chain.add(_evidence_from_tool_result(call.name, call.input, result))
            tool_results_content.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": tool_results_content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            if text.strip():
                return text

    # Final synthesis pass, explicitly asking for the structured answer.
    synth_prompt = SYNTHESIS_TEMPLATE.format(
        evidence_json=json.dumps(chain.as_llm_context(), default=str, indent=2),
        question=question,
    )
    final = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": synth_prompt}],
    )
    return "".join(b.text for b in final.content if b.type == "text")


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (used for both OpenAI and Ollama, which exposes
# an OpenAI-compatible API at /v1) — tool-calling support on Ollama depends
# on the model (e.g. llama3.1, qwen2.5, mistral-nemo support it; smaller/older
# models may not).
# ---------------------------------------------------------------------------
def _run_openai_compatible_investigation(
    question: str, plan: dict, chain: EvidenceChain,
    api_key: str, model: str, base_url: Optional[str] = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    tools = get_openai_tool_schemas()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INVESTIGATION_STARTER_TEMPLATE.format(
            question=question, intent=plan["intent"], suggested_tools=plan["suggested_tools"],
        )},
    ]

    for step in range(config.MAX_INVESTIGATION_STEPS):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            if msg.content and msg.content.strip():
                return msg.content
            break

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            chain.add_trace(f"Called `{tc.function.name}` with {json.dumps(args)}")
            result = call_tool(tc.function.name, args)
            chain.add(_evidence_from_tool_result(tc.function.name, args, result))
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    synth_prompt = SYNTHESIS_TEMPLATE.format(
        evidence_json=json.dumps(chain.as_llm_context(), default=str, indent=2),
        question=question,
    )
    final = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": synth_prompt}],
    )
    return final.choices[0].message.content or ""


def _run_openai_investigation(question: str, plan: dict, chain: EvidenceChain) -> str:
    return _run_openai_compatible_investigation(
        question, plan, chain, api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL,
    )


def _run_ollama_investigation(question: str, plan: dict, chain: EvidenceChain) -> str:
    return _run_openai_compatible_investigation(
        question, plan, chain, api_key="ollama", model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Deterministic fallback (no LLM available) — MASTER PROMPT #37
# ---------------------------------------------------------------------------
def _run_fallback_investigation(question: str, plan: dict, chain: EvidenceChain) -> str:
    """
    Executes the planner's suggested tools directly (no LLM reasoning between
    steps) and formats the raw evidence into the standard response template.
    Less adaptive than the LLM path, but keeps the product fully functional
    if the LLM API is down or unconfigured.
    """
    for tool_name in plan["suggested_tools"]:
        if tool_name not in TOOL_REGISTRY:
            continue
        args: dict = {}
        # Tools that require dates or a user_id we don't have are skipped in
        # fallback mode rather than guessed.
        func, schema = TOOL_REGISTRY[tool_name]
        required = schema["input_schema"].get("required", [])
        if any(r not in args for r in required):
            continue
        chain.add_trace(f"Called `{tool_name}` (fallback mode, no arguments inferred)")
        result = call_tool(tool_name, args)
        chain.add(_evidence_from_tool_result(tool_name, args, result))

    if chain.is_empty():
        return (
            "I couldn't run an automatic investigation in fallback mode because "
            "this question needs specific parameters (like a date range or user ID) "
            "that I can't infer without an LLM connected. Please configure an LLM "
            "provider, or use the dashboard filters to narrow your question."
        )

    lines = ["🔎 Investigation Summary", "Ran available analytics tools in deterministic fallback mode.", "",
              "📊 Evidence"]
    for ev in chain.items:
        lines.append(f"- {ev.source_tool}: {json.dumps(ev.value, default=str)}")
    lines += ["", "💡 Recommendation", "Connect an LLM provider for adaptive, multi-step investigation "
                                       "and a synthesized root-cause explanation."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def investigate(question: str) -> InvestigationResult:
    """
    Main entry point called by the Streamlit app. Runs the full investigation
    loop and returns a structured result (never raises).
    """
    plan = build_initial_plan(question)
    chain = EvidenceChain()

    try:
        if config.LLM_PROVIDER == "anthropic" and config.ANTHROPIC_API_KEY:
            answer = _run_anthropic_investigation(question, plan, chain)
            mode = "llm"
        elif config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
            answer = _run_openai_investigation(question, plan, chain)
            mode = "llm"
        elif config.LLM_PROVIDER == "ollama":
            answer = _run_ollama_investigation(question, plan, chain)
            mode = "llm"
        else:
            answer = _run_fallback_investigation(question, plan, chain)
            mode = "fallback"

        return InvestigationResult(
            question=question, intent=plan["intent"], trace=chain.trace,
            evidence=chain.as_llm_context(), answer=answer, mode=mode,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Investigation failed, falling back to deterministic mode")
        try:
            fallback_chain = EvidenceChain()
            answer = _run_fallback_investigation(question, plan, fallback_chain)
            return InvestigationResult(
                question=question, intent=plan["intent"], trace=fallback_chain.trace,
                evidence=fallback_chain.as_llm_context(), answer=answer, mode="fallback",
                error=f"LLM investigation failed ({e}); used fallback mode.",
            )
        except Exception as e2:  # noqa: BLE001
            return InvestigationResult(
                question=question, intent=plan.get("intent", "general_data_question"),
                answer="I couldn't complete this investigation due to an internal error. "
                       "Please try rephrasing your question or check the dataset is loaded.",
                mode="error", error=str(e2),
            )
