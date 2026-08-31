import asyncio
import re

import httpx
from app.core.config import settings

# Free LLM tiers rate-limit aggressively (Groq's is 8k tokens/minute), and a
# ReAct loop re-sends its whole accumulated context every iteration, so it hits
# that ceiling mid-run. A 429 partway through means the agent throws away the
# retrieval it already paid for, so it is worth waiting out rather than failing.
_MAX_ATTEMPTS = 4


async def _sleep_for_retry(response: httpx.Response, attempt: int) -> None:
    """
    Wait before retrying a rate-limited call.

    Prefers the server's own advice — Groq returns Retry-After, and its error
    body says e.g. "try again in 10.11s" — because guessing shorter just burns
    another attempt. Falls back to exponential backoff.
    """
    delay = None

    header = response.headers.get("retry-after")
    if header:
        try:
            delay = float(header)
        except ValueError:
            delay = None

    if delay is None:
        match = re.search(r"try again in ([0-9.]+)s", response.text)
        if match:
            delay = float(match.group(1))

    # +1s of margin: the window has to have actually rolled over, not just
    # nearly rolled over, or the retry burns an attempt for nothing.
    await asyncio.sleep((delay + 1.0) if delay is not None else 2**attempt)

class LLMService:
    """
    Central service for querying Large Language Models.
    Abstracts API endpoints so you can switch between Gemini, OpenAI, and Ollama.
    """
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.api_key = settings.llm_api_key
        self.api_base = settings.llm_api_base
        self.model = settings.llm_model

    async def generate_response(self, system_instruction: str, prompt: str) -> str:
        """
        Generates a text completion given a system instruction and user prompt.
        """
        if self.provider == "gemini":
            return await self._call_gemini(system_instruction, prompt)
        elif self.provider == "ollama":
            return await self._call_ollama(system_instruction, prompt)
        else:
            return await self._call_openai_compatible(system_instruction, prompt)

    async def _call_gemini(self, system_instruction: str, prompt: str) -> str:
        # Gemini-specific URL and payload structure
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": 2048
            }
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Gemini API Error: {response.text}")
            
            data = response.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                raise Exception(f"Unexpected response structure from Gemini API: {data}")

    async def _call_ollama(self, system_instruction: str, prompt: str) -> str:
        """
        Queries a local Ollama instance using its native /api/chat endpoint.
        """
        url = f"{self.api_base}/api/chat"
        headers = {"Content-Type": "application/json"}

        # Ollama expects models, system messages, and parameters in this structure
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "stream": False,  # We need the full answer at once to parse reasoning tokens
            "options": {
                "temperature": settings.llm_temperature
            }
        }

        async with httpx.AsyncClient(timeout=60.0) as client:  # Local models can take longer to reply
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Ollama API Error ({response.status_code}): {response.text}")
            
            data = response.json()
            try:
                return data["message"]["content"]
            except KeyError:
                raise Exception(f"Unexpected response structure from Ollama: {data}")

    async def _call_openai_compatible(self, system_instruction: str, prompt: str) -> str:
        # Standard OpenAI API structure
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": settings.llm_temperature
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(_MAX_ATTEMPTS):
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

                # 429 = rate limited, 5xx = transient upstream fault. Anything
                # else (bad key, unknown model) will not improve on a retry.
                if response.status_code != 429 and response.status_code < 500:
                    raise Exception(f"LLM API Error: {response.text}")

                if attempt == _MAX_ATTEMPTS - 1:
                    raise Exception(f"LLM API Error after {_MAX_ATTEMPTS} attempts: {response.text}")

                await _sleep_for_retry(response, attempt)
