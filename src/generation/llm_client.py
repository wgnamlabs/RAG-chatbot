"""
llm_client.py — Client giao tiếp với Ollama local LLM.

Model mặc định: qwen2.5:7b (dùng cho cả generation và query rewriting)
  - Chạy local qua Ollama, không cần API key, không phát sinh chi phí
  - temperature=0.0 để output nhất quán
  - Hỗ trợ streaming để trải nghiệm tốt hơn khi demo

Yêu cầu: Ollama đang chạy ở http://localhost:11434
         ollama pull qwen2.5:7b
"""

from dataclasses import dataclass
from typing import Generator, List, Optional


@dataclass
class LLMConfig:
    """Cấu hình cho OllamaLLMClient.

    Attributes:
        model:       Tên model Ollama (mặc định qwen2.5:7b).
        base_url:    URL Ollama server.
        temperature: Độ ngẫu nhiên (0.0 = deterministic).
        max_tokens:  Số token tối đa cho output (-1 = không giới hạn).
        timeout:     Timeout HTTP (giây).
        stream:      Bật streaming (True cho UI, False cho batch eval).
    """
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 120
    stream: bool = False


class OllamaLLMClient:
    """Client gọi Ollama local LLM qua REST API.

    Supports both streaming và non-streaming mode.
    Interface generic: có thể đổi model qua config.

    Ví dụ:
        client = OllamaLLMClient()
        answer = client.chat(messages)          # non-streaming
        for chunk in client.chat_stream(messages):  # streaming
            print(chunk, end="", flush=True)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    def _check_model_available(self) -> bool:
        """Kiểm tra model đã được pull chưa."""
        try:
            import requests
            resp = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10,
            )
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                model_base = self.config.model.split(":")[0]
                return any(model_base in m for m in models)
        except Exception:
            pass
        return False

    def check_connection(self) -> bool:
        """Kiểm tra Ollama server có đang chạy không."""
        try:
            import requests
            resp = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(self, messages: List[dict]) -> str:
        """Gọi LLM và trả về toàn bộ câu trả lời (non-streaming).

        Args:
            messages: Danh sách dict {"role": str, "content": str}.

        Returns:
            Câu trả lời hoàn chỉnh dạng string.

        Raises:
            ConnectionError: Nếu Ollama không phản hồi.
            RuntimeError:    Nếu API trả về lỗi.
        """
        import requests

        payload = {
            "model":   self.config.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Không kết nối được Ollama tại {self.config.base_url}. "
                "Hãy chạy `ollama serve` và đảm bảo model đã được pull."
            )

        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def chat_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        """Gọi LLM theo chế độ streaming, yield từng token/chunk.

        Args:
            messages: Danh sách dict {"role": str, "content": str}.

        Yields:
            Từng đoạn text (chunk) khi Ollama trả về.
        """
        import requests
        import json as _json

        payload = {
            "model":    self.config.model,
            "messages": messages,
            "stream":   True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            with requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.config.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = _json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done", False):
                            break
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Không kết nối được Ollama tại {self.config.base_url}."
            )
