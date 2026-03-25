"""
Teacher LLM provider abstraction.

Supported providers:
  local_hf   — local Hugging Face model (default, Qwen3-8B)
  vllm       — local vLLM OpenAI-compatible server
  openai     — OpenAI API
  openrouter — OpenRouter API (openai-compatible)
  azure      — Azure OpenAI
"""
from __future__ import annotations

import json
import re
import threading
import os
import time
from typing import Any, Optional, Dict, List

try:
    import httpx
except ImportError:
    httpx = None


class _ChatCompletionsResponse:
    """Minimal duck-type of openai ChatCompletion response."""

    class _Choice:
        class _Message:
            def __init__(self, content: str):
                self.content = content

        def __init__(self, content: str):
            self.message = self._Message(content)

    def __init__(self, content: str):
        self.choices = [self._Choice(content)]


class RemoteHTTPXClient:
    """
    Client for remote APIs (OpenRouter, Azure, DashScope) using httpx directly.
    Implements the same .chat.completions.create interface as LocalHFClient.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        endpoint: str,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        if httpx is None:
            raise ImportError("httpx is required for remote providers. Install it with `pip install httpx`.")

        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.timeout = timeout

        self.chat = _RemoteChatNamespace(self)

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        endpoint_lower = self.endpoint.lower()
        api_key = self.api_key

        if "openrouter" in endpoint_lower:
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "https://github.com/serkan/llm_training")
                headers["X-Title"] = os.getenv("OPENROUTER_SITE_TITLE", "LLM Training Data Gen")
        elif "azure" in endpoint_lower or "cognitiveservices" in endpoint_lower:
            if api_key:
                # User requested both Bearer and api-key for Azure
                headers["Authorization"] = f"Bearer {api_key}"
                headers["api-key"] = api_key
        elif "dashscope" in endpoint_lower:
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        elif api_key:
            # Fallback for generic OpenAI-compatible
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def check_health(self) -> bool:
        """
        Perform a provider-specific health check.
        Returns True if healthy/reachable, False otherwise.
        """
        try:
            headers = self._build_headers()
            with httpx.Client(timeout=10.0) as client:
                if self.provider == "dashscope":
                    # DashScope 'ping' check - simulate with minimal generation
                    # User snippet used a specific embedding endpoint, but we are a chat client.
                    # We'll use the chat endpoint with a dummy message.
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1
                    }
                    response = client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    return True

                elif self.provider == "azure":
                    # Azure 'list models' check
                    # We need to extract the base URL from the full chat endpoint
                    # Endpoint: {base}/openai/deployments/{model}/chat/completions?api-version={ver}
                    # Target:   {base}/openai/models?api-version={ver}
                    
                    # Regex to capture up to /openai
                    match = re.match(r"(.*?/openai)", self.endpoint)
                    if match:
                        base_openai = match.group(1)
                        # Extract api-version
                        api_ver = "2024-02-01"
                        if "api-version=" in self.endpoint:
                            api_ver = self.endpoint.split("api-version=")[1].split("&")[0]
                        
                        check_url = f"{base_openai}/models?api-version={api_ver}"
                        response = client.get(check_url, headers=headers)
                        response.raise_for_status()
                        return True

                # Default/OpenRouter check: simple generation
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1
                }
                response = client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return True

        except Exception as e:
            print(f"[{self.provider}] Health check failed: {e}")
            return False

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        response_format: Optional[Dict] = None,
    ) -> str:
        headers = self._build_headers()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format and response_format.get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    # Azure needs specific deployment URL construction if not provided fully
                    # But we assume self.endpoint is the full chat completions URL
                    response = client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()

                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
            except Exception as e:
                print(f"[{self.provider}] Request failed (attempt {attempt+1}/{self.max_retries + 1}): {e}")
                last_error = e
                time.sleep(1 * (2 ** attempt))  # Exponential backoff

        raise last_error or Exception("Unknown error in RemoteHTTPXClient")


class _RemoteCompletionsNamespace:
    def __init__(self, client: RemoteHTTPXClient):
        self._client = client

    def create(
        self,
        model: str,  # ignored, uses client.model
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> _ChatCompletionsResponse:
        content = self._client._generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return _ChatCompletionsResponse(content)


class _RemoteChatNamespace:
    def __init__(self, client: RemoteHTTPXClient):
        self.completions = _RemoteCompletionsNamespace(client)


class LocalHFClient:
    """
    Drop-in replacement for the OpenAI client that runs a local
    Hugging Face causal-LM with chat-template support.

    Usage mirrors openai.OpenAI():
        client.chat.completions.create(model=..., messages=..., ...)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        quantization: str = "4bit",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        torch_dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._quantization = quantization
        self._device_map = device_map
        self._torch_dtype = torch_dtype

        self.chat = _ChatNamespace(self)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[LocalHFClient] Loading {self.model_name} ({self._quantization}) ...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        bnb_config = None
        if self._quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self._quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        dtype = getattr(torch, self._torch_dtype, torch.bfloat16)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self._device_map,
            trust_remote_code=True,
            dtype=dtype,
        )
        self._model.eval()
        print(f"[LocalHFClient] Model loaded.")

    def _generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 512,
        response_format: Optional[dict] = None,
    ) -> str:
        self._ensure_loaded()
        import torch

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Qwen3: inject /no_think into the last user message to disable thinking
        if "Qwen3" in self.model_name or "qwen3" in self.model_name.lower():
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages = [m.copy() for m in messages]
                    messages[i]["content"] = messages[i]["content"] + " /no_think"
                    break
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        max_new = min(max_tokens, self.max_new_tokens)

        with self._lock:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    repetition_penalty=1.2,  # Strong penalty to prevent 0.6B loops
                )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip any residual think tokens (full blocks or orphan tags)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"</think>", "", raw)
        raw = re.sub(r"<think>", "", raw)
        raw = raw.strip()

        if response_format and response_format.get("type") == "json_object":
            raw = _force_json(raw)

        return raw


class _CompletionsNamespace:
    def __init__(self, client: LocalHFClient):
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 512,
        response_format: Optional[dict] = None,
        **kwargs,
    ) -> _ChatCompletionsResponse:
        content = self._client._generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return _ChatCompletionsResponse(content)


class _ChatNamespace:
    def __init__(self, client: LocalHFClient):
        self.completions = _CompletionsNamespace(client)


def _force_json(text: str) -> str:
    """Best-effort extraction of a JSON object from raw model output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # Try to repair common issues: trailing commas, single quotes
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = fixed.replace("'", '"')
    match2 = re.search(r"\{.*\}", fixed, re.DOTALL)
    if match2:
        try:
            json.loads(match2.group())
            return match2.group()
        except json.JSONDecodeError:
            pass
    return text


def build_teacher_pool(cfg: dict) -> list:
    """
    Build a list of provider clients from config.

    Reads ``cfg["teachers"]`` (a list of provider configs) if present,
    otherwise falls back to the single ``cfg["teacher"]`` entry.

    Example YAML::

        teachers:
          - provider: local_hf
            model: Qwen/Qwen3-8B
            quantization: 4bit
          - provider: azure
            model: gpt-4o-mini
            api_key: AZURE_OPENAI_API_KEY
            azure_endpoint: AZURE_OPENAI_ENDPOINT
            api_version: "2024-02-01"
          - provider: openrouter
            model: qwen/qwen3-8b
            api_key: OPENROUTER_API_KEY

    Returns a list of client objects, one per entry.  The caller should
    create one ``Labeler`` / ``TurnGenerator`` per client and distribute
    episodes across them with a ``ThreadPoolExecutor``.
    """
    teachers_cfg = cfg.get("teachers")
    if teachers_cfg:
        clients = []
        for tc in teachers_cfg:
            print(f"[TeacherPool] Building client: provider={tc.get('provider')}  model={tc.get('model', '')}")
            clients.append(build_teacher_client(tc))
        return clients
    # Backwards-compatible single-provider path
    return [build_teacher_client(cfg["teacher"])]


def build_teacher_client(teacher_cfg: dict) -> Any:
    """
    Factory — returns a client with a .chat.completions.create() interface.

    teacher_cfg keys:
      provider     : local_hf | vllm | openai | openrouter | azure
      model        : model name / deployment
      quantization : 4bit | 8bit | none   (local_hf only)
      device_map   : auto | cuda | cpu    (local_hf only)
      api_key      : env var name or literal key (non-local providers)
      base_url     : override endpoint (vllm / openrouter)
      api_version  : Azure API version    (azure only)
      azure_endpoint: Azure endpoint URL  (azure only)
    """
    import os

    provider = teacher_cfg.get("provider", "local_hf")

    if provider == "local_hf":
        return LocalHFClient(
            model_name=teacher_cfg.get("model", "Qwen/Qwen3-8B"),
            quantization=teacher_cfg.get("quantization", "4bit"),
            device_map=teacher_cfg.get("device_map", "auto"),
            max_new_tokens=teacher_cfg.get("max_tokens", 512),
            torch_dtype=teacher_cfg.get("torch_dtype", "bfloat16"),
        )

    if provider == "vllm":
        from openai import OpenAI
        base_url = teacher_cfg.get("base_url", "http://localhost:8000/v1")
        api_key  = _resolve_key(teacher_cfg.get("api_key", "dummy"))
        return OpenAI(api_key=api_key, base_url=base_url)

    if provider == "openai":
        from openai import OpenAI
        api_key = _resolve_key(teacher_cfg.get("api_key", "OPENAI_API_KEY"))
        return OpenAI(api_key=api_key)

    if provider == "openrouter":
        api_key  = _resolve_key(teacher_cfg.get("api_key", "OPENROUTER_API_KEY"))
        base_url = teacher_cfg.get("base_url", "https://openrouter.ai/api/v1")
        # Ensure base_url doesn't end with slash to avoid double slash
        base_url = base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        
        return RemoteHTTPXClient(
            provider="openrouter",
            model=teacher_cfg.get("model", "qwen/qwen3-8b"),
            api_key=api_key,
            endpoint=endpoint,
        )

    if provider == "azure":
        api_key = _resolve_key(teacher_cfg.get("api_key", "AZURE_OPENAI_API_KEY"))

        # New-style: Azure AI Foundry / inference endpoints expose an OpenAI-compatible
        # /openai/v1/ base URL.  When `base_url` is set, delegate directly to the
        # standard OpenAI SDK — no api-version needed.
        if "base_url" in teacher_cfg:
            from openai import OpenAI
            base_url = teacher_cfg["base_url"]
            return OpenAI(base_url=base_url, api_key=api_key)

        # Legacy-style: construct full REST URL from azure_endpoint + deployment name.
        azure_endpoint = teacher_cfg.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        # Ensure azure_endpoint doesn't end with slash
        azure_endpoint = azure_endpoint.rstrip("/")

        model = teacher_cfg.get("model", "gpt-4o")
        api_version = teacher_cfg.get("api_version", "2024-02-01")

        # Construct full REST URL: {endpoint}/openai/deployments/{model}/chat/completions?api-version={ver}
        endpoint = f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"

        return RemoteHTTPXClient(
            provider="azure",
            model=model,
            api_key=api_key,
            endpoint=endpoint,
        )

    if provider == "dashscope":
        api_key = _resolve_key(teacher_cfg.get("api_key", "DASHSCOPE_API_KEY"))
        # DashScope endpoint is usually specific, e.g. https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
        # But for OpenAI-compatible interface it might be different. 
        # Assuming standard compatible endpoint or user provides full URL.
        base_url = teacher_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        base_url = base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        
        return RemoteHTTPXClient(
            provider="dashscope",
            model=teacher_cfg.get("model", "qwen-turbo"),
            api_key=api_key,
            endpoint=endpoint,
        )

    raise ValueError(
        f"Unknown teacher provider: '{provider}'. "
        "Choose from: local_hf, vllm, openai, openrouter, azure, dashscope"
    )


def _resolve_key(key_or_env: str) -> str:
    """If value looks like an env-var name (uppercase + underscores), resolve it."""
    import os
    if key_or_env and re.match(r"^[A-Z][A-Z0-9_]+$", key_or_env):
        resolved = os.environ.get(key_or_env, "")
        if not resolved:
            raise EnvironmentError(
                f"Environment variable '{key_or_env}' is not set. "
                "Export it before running the pipeline."
            )
        return resolved
    return key_or_env
