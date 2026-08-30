"""
DataLens AI — Agent Orchestrator

Implements the investigation loop:

    Question -> Intent -> Plan -> Tool Call -> Evidence -> Evaluate
    -> More evidence needed? -> Another tool
    -> Synthesize -> Answer

Model-agnostic:
- Anthropic tool use
- OpenAI function calling
- Ollama OpenAI-compatible API
- Deterministic fallback when no LLM is available

The Ollama path additionally handles models that return tool calls as
JSON text instead of native OpenAI tool_calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import config

from src.agent.evidence import Evidence, EvidenceChain
from src.agent.planner import build_initial_plan
from src.agent.prompts import (
    SYSTEM_PROMPT,
    INVESTIGATION_STARTER_TEMPLATE,
    SYNTHESIS_TEMPLATE,
)
from src.agent.tools import (
    TOOL_REGISTRY,
    get_tool_schemas,
    get_openai_tool_schemas,
    call_tool,
)

logger = logging.getLogger(__name__)


# ============================================================================
# RESULT OBJECT
# ============================================================================

@dataclass
class InvestigationResult:
    question: str
    intent: str
    trace: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    answer: str = ""
    charts_needed: list[str] = field(default_factory=list)
    mode: str = "llm"
    error: Optional[str] = None


# ============================================================================
# EVIDENCE
# ============================================================================

def _evidence_from_tool_result(
    tool_name: str,
    args: dict,
    result: dict,
) -> Evidence:
    """
    Convert a tool result into an Evidence object.
    """

    return Evidence(
        metric=tool_name,
        value=result.get("result"),
        source_tool=tool_name,
        raw_result={
            "args": args,
            **result,
        },
    )


# ============================================================================
# OLLAMA / OPENAI JSON TOOL-CALL PARSER
# ============================================================================

def _extract_text_tool_call(
    content: str,
) -> tuple[str, dict] | None:
    """
    Detect a tool call when the model returns it as normal text.

    Some Ollama models occasionally produce:

        {
            "name": "analyze_merchants",
            "parameters": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "top_n": 10
            }
        }

    instead of using OpenAI's native:

        message.tool_calls

    This function converts the text representation into:

        ("analyze_merchants", {...})
    """

    if not content:
        return None

    text = content.strip()

    # ------------------------------------------------------------
    # Remove markdown code fences
    # ------------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # ------------------------------------------------------------
    # Find JSON object inside explanatory text
    # ------------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    json_text = text[start:end + 1]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # ------------------------------------------------------------
    # Tool name
    # ------------------------------------------------------------

    name = data.get("name")

    if not isinstance(name, str):
        return None

    # ------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------

    arguments = data.get(
        "parameters",
        data.get("arguments", {}),
    )

    if not isinstance(arguments, dict):
        return None

    return name, arguments


# ============================================================================
# ANTHROPIC BACKEND
# ============================================================================

def _run_anthropic_investigation(
    question: str,
    plan: dict,
    chain: EvidenceChain,
) -> str:

    import anthropic

    client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY
    )

    tools = get_tool_schemas()

    messages = [
        {
            "role": "user",
            "content": INVESTIGATION_STARTER_TEMPLATE.format(
                question=question,
                intent=plan["intent"],
                suggested_tools=plan["suggested_tools"],
            ),
        }
    ]

    for step in range(config.MAX_INVESTIGATION_STEPS):

        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        tool_calls = [
            block
            for block in response.content
            if block.type == "tool_use"
        ]

        # ------------------------------------------------------------
        # Model answered without requesting a tool
        # ------------------------------------------------------------

        if not tool_calls:

            text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            if text.strip():
                return text

            break

        # ------------------------------------------------------------
        # Execute tools
        # ------------------------------------------------------------

        tool_results_content = []

        for call in tool_calls:

            args = call.input or {}

            chain.add_trace(
                f"Called `{call.name}` with "
                f"{json.dumps(args, default=str)}"
            )

            result = call_tool(
                call.name,
                args,
            )

            chain.add(
                _evidence_from_tool_result(
                    call.name,
                    args,
                    result,
                )
            )

            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(
                        result,
                        default=str,
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": tool_results_content,
            }
        )

        if response.stop_reason != "tool_use":

            text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            if text.strip():
                return text

    # ------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------

    synth_prompt = SYNTHESIS_TEMPLATE.format(
        evidence_json=json.dumps(
            chain.as_llm_context(),
            default=str,
            indent=2,
        ),
        question=question,
    )

    final = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": synth_prompt,
            }
        ],
    )

    return "".join(
        block.text
        for block in final.content
        if block.type == "text"
    )


# ============================================================================
# OPENAI / OLLAMA BACKEND
# ============================================================================

def _run_openai_compatible_investigation(
    question: str,
    plan: dict,
    chain: EvidenceChain,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> str:

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or "not-needed",
        base_url=base_url,
    )

    tools = get_openai_tool_schemas()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": INVESTIGATION_STARTER_TEMPLATE.format(
                question=question,
                intent=plan["intent"],
                suggested_tools=plan["suggested_tools"],
            ),
        },
    ]

    # ========================================================================
    # INVESTIGATION LOOP
    # ========================================================================

    for step in range(config.MAX_INVESTIGATION_STEPS):

        logger.info(
            "Investigation step %s/%s using %s",
            step + 1,
            config.MAX_INVESTIGATION_STEPS,
            model,
        )

        try:

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )

        except Exception:

            logger.exception(
                "LLM request failed at investigation step %s",
                step + 1,
            )

            raise

        msg = response.choices[0].message

        # ====================================================================
        # CASE 1 — NORMAL NATIVE TOOL CALL
        # ====================================================================

        if msg.tool_calls:

            messages.append(
                msg.model_dump(
                    exclude_none=True
                )
            )

            for tc in msg.tool_calls:

                tool_name = tc.function.name

                try:

                    args = json.loads(
                        tc.function.arguments or "{}"
                    )

                except json.JSONDecodeError as exc:

                    logger.warning(
                        "Invalid JSON tool arguments for %s: %s",
                        tool_name,
                        exc,
                    )

                    args = {}

                chain.add_trace(
                    f"Called `{tool_name}` with "
                    f"{json.dumps(args, default=str)}"
                )

                result = call_tool(
                    tool_name,
                    args,
                )

                chain.add(
                    _evidence_from_tool_result(
                        tool_name,
                        args,
                        result,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            result,
                            default=str,
                        ),
                    }
                )

            # Continue investigation after tool execution.
            continue

        # ====================================================================
        # CASE 2 — OLLAMA RETURNS TOOL CALL AS NORMAL JSON TEXT
        # ====================================================================

        content = msg.content or ""

        parsed_call = _extract_text_tool_call(
            content
        )

        if parsed_call:

            tool_name, args = parsed_call

            # ---------------------------------------------------------------
            # Validate tool name
            # ---------------------------------------------------------------

            if tool_name in TOOL_REGISTRY:

                logger.info(
                    "Detected text-based tool call: %s",
                    tool_name,
                )

                chain.add_trace(
                    f"Called `{tool_name}` with "
                    f"{json.dumps(args, default=str)}"
                )

                result = call_tool(
                    tool_name,
                    args,
                )

                chain.add(
                    _evidence_from_tool_result(
                        tool_name,
                        args,
                        result,
                    )
                )

                # -----------------------------------------------------------
                # Tell the model the tool was executed
                # -----------------------------------------------------------

                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool `{tool_name}` has been executed.\n\n"
                            "Actual tool result:\n"
                            f"{json.dumps(result, default=str)}\n\n"
                            "Use this evidence to continue the investigation. "
                            "If more evidence is required, call another "
                            "appropriate tool. Otherwise provide the final "
                            "structured answer."
                        ),
                    }
                )

                continue

            # ---------------------------------------------------------------
            # Unknown tool
            # ---------------------------------------------------------------

            logger.warning(
                "Model requested unknown tool: %s",
                tool_name,
            )

        # ====================================================================
        # CASE 3 — NORMAL TEXT ANSWER
        # ====================================================================

        if content.strip():

            return content

        break

    # ========================================================================
    # FINAL SYNTHESIS
    # ========================================================================

    synth_prompt = SYNTHESIS_TEMPLATE.format(
        evidence_json=json.dumps(
            chain.as_llm_context(),
            default=str,
            indent=2,
        ),
        question=question,
    )

    final = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": synth_prompt,
            },
        ],
    )

    return (
        final.choices[0].message.content
        or ""
    )


# ============================================================================
# OPENAI
# ============================================================================

def _run_openai_investigation(
    question: str,
    plan: dict,
    chain: EvidenceChain,
) -> str:

    return _run_openai_compatible_investigation(
        question=question,
        plan=plan,
        chain=chain,
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL,
    )


# ============================================================================
# OLLAMA
# ============================================================================

def _run_ollama_investigation(
    question: str,
    plan: dict,
    chain: EvidenceChain,
) -> str:

    return _run_openai_compatible_investigation(
        question=question,
        plan=plan,
        chain=chain,
        api_key="ollama",
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


# ============================================================================
# DETERMINISTIC FALLBACK
# ============================================================================

def _run_fallback_investigation(
    question: str,
    plan: dict,
    chain: EvidenceChain,
) -> str:
    """
    Executes the planner's suggested tools directly without an LLM.

    This keeps DataLens functional even when no LLM is available.
    """

    for tool_name in plan["suggested_tools"]:

        if tool_name not in TOOL_REGISTRY:
            continue

        args: dict = {}

        func, schema = TOOL_REGISTRY[tool_name]

        required = schema[
            "input_schema"
        ].get(
            "required",
            [],
        )

        # Skip tools requiring parameters that cannot be inferred.
        if any(
            required_arg not in args
            for required_arg in required
        ):
            continue

        chain.add_trace(
            f"Called `{tool_name}` "
            "(fallback mode, no arguments inferred)"
        )

        result = call_tool(
            tool_name,
            args,
        )

        chain.add(
            _evidence_from_tool_result(
                tool_name,
                args,
                result,
            )
        )

    # ========================================================================
    # Nothing could be executed
    # ========================================================================

    if chain.is_empty():

        return (
            "I couldn't run an automatic investigation in fallback mode "
            "because this question needs specific parameters such as a "
            "date range or user ID that I can't infer without an LLM "
            "connected. Please configure an LLM provider, or use the "
            "dashboard filters to narrow your question."
        )

    # ========================================================================
    # Format fallback response
    # ========================================================================

    lines = [
        "🔎 Investigation Summary",
        "Ran available analytics tools in deterministic fallback mode.",
        "",
        "📊 Evidence",
    ]

    for ev in chain.items:

        lines.append(
            f"- {ev.source_tool}: "
            f"{json.dumps(ev.value, default=str)}"
        )

    lines.extend(
        [
            "",
            "💡 Recommendation",
            (
                "Connect an LLM provider for adaptive, multi-step "
                "investigation and a synthesized root-cause explanation."
            ),
        ]
    )

    return "\n".join(lines)


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def investigate(
    question: str,
) -> InvestigationResult:
    """
    Main entry point called by Streamlit.

    The function never intentionally allows an LLM failure to crash the app.
    It falls back to deterministic analytics when possible.
    """

    plan = build_initial_plan(
        question
    )

    chain = EvidenceChain()

    try:

        # ====================================================================
        # ANTHROPIC
        # ====================================================================

        if (
            config.LLM_PROVIDER == "anthropic"
            and config.ANTHROPIC_API_KEY
        ):

            answer = _run_anthropic_investigation(
                question,
                plan,
                chain,
            )

            mode = "llm"

        # ====================================================================
        # OPENAI
        # ====================================================================

        elif (
            config.LLM_PROVIDER == "openai"
            and config.OPENAI_API_KEY
        ):

            answer = _run_openai_investigation(
                question,
                plan,
                chain,
            )

            mode = "llm"

        # ====================================================================
        # OLLAMA
        # ====================================================================

        elif config.LLM_PROVIDER == "ollama":

            answer = _run_ollama_investigation(
                question,
                plan,
                chain,
            )

            mode = "llm"

        # ====================================================================
        # NO LLM
        # ====================================================================

        else:

            answer = _run_fallback_investigation(
                question,
                plan,
                chain,
            )

            mode = "fallback"

        # ====================================================================
        # SUCCESS
        # ====================================================================

        return InvestigationResult(
            question=question,
            intent=plan["intent"],
            trace=chain.trace,
            evidence=chain.as_llm_context(),
            answer=answer,
            mode=mode,
        )

    # ========================================================================
    # LLM ERROR → FALLBACK
    # ========================================================================

    except Exception as e:

        logger.exception(
            "Investigation failed. "
            "Falling back to deterministic mode."
        )

        try:

            fallback_chain = EvidenceChain()

            answer = _run_fallback_investigation(
                question,
                plan,
                fallback_chain,
            )

            return InvestigationResult(
                question=question,
                intent=plan["intent"],
                trace=fallback_chain.trace,
                evidence=fallback_chain.as_llm_context(),
                answer=answer,
                mode="fallback",
                error=(
                    f"LLM investigation failed ({e}); "
                    "used fallback mode."
                ),
            )

        except Exception as e2:

            logger.exception(
                "Fallback investigation also failed."
            )

            return InvestigationResult(
                question=question,
                intent=plan.get(
                    "intent",
                    "general_data_question",
                ),
                answer=(
                    "I couldn't complete this investigation due to "
                    "an internal error. Please try rephrasing your "
                    "question or check that the dataset is loaded."
                ),
                mode="error",
                error=str(e2),
            )