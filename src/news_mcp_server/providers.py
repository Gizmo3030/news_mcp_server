from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, TypedDict

from pydantic import BaseModel

# Optional imports guarded at runtime


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class ChatResponse(BaseModel):
    provider: str
    model: str
    content: str
    finish_reason: Optional[str] = None


ProviderName = Literal["openai", "gemini", "ollama"]


@dataclass
class ProviderConfig:
    provider: ProviderName = "openai"
    model: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @staticmethod
    def from_env() -> "ProviderConfig":
        return ProviderConfig(
            provider=os.environ.get("LLM_PROVIDER", "openai").lower(),
            model=os.environ.get("LLM_MODEL"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )


class LLMClient:
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    async def chat(self, messages: Sequence[ChatMessage], model: Optional[str] = None) -> ChatResponse:  # noqa: D401
        """Send chat messages to the provider and return the response."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - import-time
            raise RuntimeError("openai package is required for OpenAI provider") from e
        if not cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider")
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)

    async def chat(self, messages: Sequence[ChatMessage], model: Optional[str] = None) -> ChatResponse:
        use_model = model or self.cfg.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = await self._client.chat.completions.create(
            model=use_model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=0.7,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        return ChatResponse(provider="openai", model=use_model, content=content, finish_reason=choice.finish_reason)


class GeminiClient(LLMClient):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("google-generativeai package is required for Gemini provider") from e
        if not cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini provider")
        genai.configure(api_key=cfg.gemini_api_key)
        self._genai = genai

    async def chat(self, messages: Sequence[ChatMessage], model: Optional[str] = None) -> ChatResponse:
        # google-generativeai is sync; wrap minimally using anyio's to_thread if needed.
        import anyio

        use_model = model or self.cfg.model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        gen_model = self._genai.GenerativeModel(use_model)

        def _run_sync():
            # Convert messages to a single prompt; Gemini supports multi-turn via chat, but simple join works.
            parts: List[str] = []
            for m in messages:
                prefix = {
                    "system": "[system]",
                    "user": "[user]",
                    "assistant": "[assistant]",
                }[m["role"]]
                parts.append(f"{prefix} {m['content']}")
            prompt = "\n".join(parts)
            r = gen_model.generate_content(prompt)
            return r

        r = await anyio.to_thread.run_sync(_run_sync)
        text = getattr(r, "text", None) or (r.candidates[0].content.parts[0].text if r.candidates else "")
        return ChatResponse(provider="gemini", model=use_model, content=text, finish_reason=None)


class OllamaClient(LLMClient):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        try:
            import ollama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("ollama package is required for Ollama provider") from e
        self._ollama = ollama
        self._base_url = cfg.ollama_base_url
        # Prefer a client bound to host to avoid relying solely on env
        try:
            self._client = ollama.Client(host=self._base_url)
        except Exception:
            # Fallback to env var if needed
            os.environ.setdefault("OLLAMA_HOST", self._base_url)
            self._client = None

    async def chat(self, messages: Sequence[ChatMessage], model: Optional[str] = None) -> ChatResponse:
        import anyio

        use_model = model or self.cfg.model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

        def _run_sync():
            api = self._client if self._client is not None else self._ollama
            res = api.chat(
                model=use_model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                options={"temperature": 0.7},
                keep_alive=True,
            )
            return res

        res = await anyio.to_thread.run_sync(_run_sync)
        content = res.get("message", {}).get("content", "")
        return ChatResponse(provider="ollama", model=use_model, content=content, finish_reason=None)


def make_client(cfg: Optional[ProviderConfig] = None) -> LLMClient:
    cfg = cfg or ProviderConfig.from_env()
    provider = cfg.provider
    if provider == "openai":
        return OpenAIClient(cfg)
    if provider == "gemini":
        return GeminiClient(cfg)
    if provider == "ollama":
        return OllamaClient(cfg)
    raise ValueError(f"Unknown provider: {provider}")
