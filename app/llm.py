"""LLM provider abstraction.

Single interface: generate(system, user) -> str.
No streaming on v1 — streaming is where provider SDKs diverge sharply.

To add a new provider:
  1. Implement LLMClient.generate
  2. Register in get_llm_client
"""
from abc import ABC, abstractmethod
import os


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Single-turn generation. Returns the assistant's text reply."""
        ...


class GroqClient(LLMClient):
    def __init__(self) -> None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,  # low — we want grounded, not creative
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        # Note: Anthropic system prompts are a top-level param, not a message
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def generate(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=800,
        )
        # Anthropic returns content as a list of blocks
        parts = [block.text for block in resp.content if block.type == "text"]
        return "".join(parts)


def get_llm_client() -> LLMClient:
    """Factory. Reads LLM_PROVIDER env var. Defaults to groq."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return GroqClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
