"""A stand-in LLM that replays fixed CodeAgent turns, so the agent loop, streaming
trace and UI can be tested offline without an API key."""

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage


class ScriptedModel(Model):
    def __init__(self, turns: list[str]):
        super().__init__(model_id="scripted")
        self.turns = list(turns)
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=turn,
                           token_usage=TokenUsage(input_tokens=100, output_tokens=20))


def code_turn(thought: str, code: str) -> str:
    return f"Thought: {thought}\n<code>\n{code}\n</code>"
