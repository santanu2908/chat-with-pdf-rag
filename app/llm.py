"""LLM provider abstraction.

Two interfaces:
  generate(system, messages) -> str      (blocking, used by /query)
  stream(system, messages)   -> Iterator (yields text chunks, used by /query/stream)

messages is a list of {"role": "user"|"assistant", "content": "..."} dicts.
For single-turn calls, pass a one-element list.

To add a new provider:
  1. Implement LLMClient.generate and LLMClient.stream
  2. Register in get_llm_client
"""
from abc import ABC, abstractmethod
from typing import Iterator, List, Dict
import os

Messages = List[Dict[str, str]]


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, messages: Messages) -> str:
        """Multi-turn generation. Returns the assistant's text reply."""
        ...

    @abstractmethod
    def stream(self, system: str, messages: Messages) -> Iterator[str]:
        """Streaming generation. Yields text chunks as they arrive."""
        ...


class GroqClient(LLMClient):
    def __init__(self) -> None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _build_messages(self, system: str, messages: Messages) -> list:
        return [{"role": "system", "content": system}] + messages

    def generate(self, system: str, messages: Messages) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(system, messages),
            temperature=0.2,  # low — we want grounded, not creative
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, messages: Messages) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(system, messages),
            temperature=0.2,
            max_tokens=800,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _build_messages(self, system: str, messages: Messages) -> list:
        return [{"role": "system", "content": system}] + messages

    def generate(self, system: str, messages: Messages) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(system, messages),
            temperature=0.2,
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, messages: Messages) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(system, messages),
            temperature=0.2,
            max_tokens=800,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        # Note: Anthropic system prompts are a top-level param, not a message
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def generate(self, system: str, messages: Messages) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )
        # Anthropic returns content as a list of blocks
        parts = [block.text for block in resp.content if block.type == "text"]
        return "".join(parts)

    def stream(self, system: str, messages: Messages) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        ) as resp:
            for text in resp.text_stream:
                yield text


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
