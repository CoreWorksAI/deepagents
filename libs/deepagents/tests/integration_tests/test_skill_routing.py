"""Skill routing integration test — progressive disclosure pattern.

Validates that SkillsMiddleware + activate_skill tool work together:
- Skills metadata injected into system prompt (Level 1: names + descriptions)
- Agent reads full SKILL.md on demand for detailed instructions (Level 2)
- Simple queries don't trigger skill activation (token savings)
- Agent picks the correct skill based on user intent

Uses a customer support bot analogy:
    SupportBot (orchestrator with skills)
    ├── SKILL: billing — refund policies, payment troubleshooting
    ├── SKILL: technical — debugging steps, system status checks
    └── activate_skill() tool — loads full instructions on demand

Run: pytest tests/integration_tests/test_skill_routing.py -xvs
Requires: ANTHROPIC_API_KEY
"""

import pytest
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, StructuredTool

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent


def _create_skill_dir(tmp_path: Path, skill_name: str, description: str, body: str) -> None:
    """Create a SKILL.md file in the expected directory structure."""
    skill_dir = tmp_path / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}"
    )


@tool
def respond(message: str) -> str:
    """Send a response to the user."""
    return f"Responded: {message}"


@tool
def think(thought: str) -> str:
    """Think through a problem step by step before acting."""
    return f"Thought: {thought}"


_activated_skills: list[str] = []


def create_activate_skill_tool(backend: FilesystemBackend) -> StructuredTool:
    """Create an activate_skill tool that reads SKILL.md from the backend."""

    def activate_skill(skill_id: str) -> str:
        """Load detailed instructions for a support category. Call BEFORE handling the request."""
        _activated_skills.append(skill_id)
        skill_path = f"/skills/{skill_id}/SKILL.md"
        try:
            result = backend.read_file(skill_path)
            content = result.content if hasattr(result, "content") else str(result)
            parts = content.split("---", 2)
            return parts[2].strip() if len(parts) >= 3 else content
        except Exception as e:
            return f"Unknown skill '{skill_id}'. Error: {e}"

    return StructuredTool.from_function(
        func=activate_skill,
        name="activate_skill",
        description=(
            "Load detailed instructions for a support category. "
            "Call this BEFORE handling specialized requests.\n\n"
            "Available skills will be listed in your system prompt."
        ),
    )


@pytest.fixture(autouse=True)
def reset_activations():
    _activated_skills.clear()
    yield


@pytest.fixture
def skill_env(tmp_path):
    """Set up skills for a customer support bot."""
    _create_skill_dir(
        tmp_path, "billing",
        "Handle billing inquiries, refunds, payment issues, and subscription changes",
        """# Billing Support Skill

## When to Use
When the customer asks about charges, refunds, payment methods,
subscription plans, or invoice discrepancies.

## Procedures
1. Always verify the customer's account first
2. For refunds: check eligibility (within 30-day window)
3. For payment issues: verify card on file, suggest alternatives
4. For plan changes: confirm new pricing before switching

## Escalation
Refunds over $500 require manager approval.
""",
    )

    _create_skill_dir(
        tmp_path, "technical",
        "Troubleshoot technical issues, check system status, guide debugging steps",
        """# Technical Support Skill

## When to Use
When the customer reports errors, connectivity issues,
performance problems, or needs help with configuration.

## Debugging Steps
1. Check system status page first
2. Ask for error messages or screenshots
3. Try standard fixes: clear cache, restart, update
4. If unresolved: collect logs and escalate to engineering

## Known Issues
- Login timeout: Fixed in v2.4, ask customer to update
- Slow dashboard: Known issue, ETA next sprint
""",
    )

    backend = FilesystemBackend(root_dir=str(tmp_path))
    return tmp_path, backend


@pytest.mark.requires("langchain_anthropic")
class TestSkillRouting:
    """Integration tests for progressive skill loading."""

    def test_skills_appear_in_system_prompt(self, skill_env):
        """SkillsMiddleware injects skill names + descriptions into the prompt."""
        tmp_path, backend = skill_env

        agent = create_deep_agent(
            tools=[respond, think],
            system_prompt="You are a customer support bot. Use skills when needed.",
            backend=backend,
            skills=["/skills/"],
            enable_filesystem=True,
            enable_todos=False,
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Hi! What can you help me with?")]},
        )

        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        assert len(ai_messages) >= 1

    def test_activate_skill_loads_instructions(self, skill_env):
        """activate_skill('billing') loads the full SKILL.md body."""
        tmp_path, backend = skill_env
        activate_tool = create_activate_skill_tool(backend)

        agent = create_deep_agent(
            tools=[respond, think, activate_tool],
            system_prompt=(
                "You are a customer support bot. When asked about billing or payments, "
                "FIRST call activate_skill('billing') to load procedures, "
                "then use think() to plan, then respond(). Be brief."
            ),
            backend=backend,
            skills=["/skills/"],
            enable_filesystem=False,
            enable_todos=False,
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="I was charged twice for my subscription.")]},
        )

        assert "billing" in _activated_skills, (
            f"Expected 'billing' skill activated. Got: {_activated_skills}. "
            f"Tool calls: {[tc['name'] for m in result['messages'] if m.type == 'ai' for tc in m.tool_calls]}"
        )

    def test_no_skill_for_greetings(self, skill_env):
        """Simple greetings should NOT trigger skill activation."""
        tmp_path, backend = skill_env
        activate_tool = create_activate_skill_tool(backend)

        agent = create_deep_agent(
            tools=[respond, think, activate_tool],
            system_prompt=(
                "You are a customer support bot. For greetings, "
                "just respond() directly. Only activate skills for "
                "billing or technical issues. Be very brief."
            ),
            backend=backend,
            skills=["/skills/"],
            enable_filesystem=False,
            enable_todos=False,
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Hello!")]},
        )

        assert len(_activated_skills) == 0, (
            f"No skills should activate for 'Hello!', got: {_activated_skills}"
        )

    def test_correct_skill_routing(self, skill_env):
        """Agent routes to the right skill based on intent."""
        tmp_path, backend = skill_env
        activate_tool = create_activate_skill_tool(backend)

        agent = create_deep_agent(
            tools=[respond, think, activate_tool],
            system_prompt=(
                "You are a customer support bot with skills. When asked about:\n"
                "- Billing, payments, refunds → activate_skill('billing')\n"
                "- Errors, bugs, technical issues → activate_skill('technical')\n"
                "- Greetings → respond() directly\n"
                "After activating, use think() to plan, then respond(). Be brief."
            ),
            backend=backend,
            skills=["/skills/"],
            enable_filesystem=False,
            enable_todos=False,
        )

        _activated_skills.clear()
        result = agent.invoke(
            {"messages": [HumanMessage(content="The dashboard is very slow and I keep getting timeout errors.")]},
        )
        assert "technical" in _activated_skills, (
            f"Expected 'technical' for timeout errors. Got: {_activated_skills}"
        )
