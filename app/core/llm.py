import httpx
from app.core.config import settings

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

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"LLM API Error: {response.text}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
