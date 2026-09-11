from abc import ABC, abstractmethod

import httpx


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):

    def generate(self, prompt: str) -> str:
        return (
            "Based on the retrieved documents, the answer is supported "
            "by the provided context. [1]"
        )


class OllamaLLMProvider(LLMProvider):

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]