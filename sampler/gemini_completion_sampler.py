import asyncio
import base64
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import types

MessageList = List[Dict[str, Any]]


def _strip_data_url(data_url: str) -> Tuple[str, bytes]:
    """
    Parse a data URL like: data:image/png;base64,xxxx
    Returns: (mime_type, raw_bytes)
    """
    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")

    header, b64 = data_url.split(",", 1)
    # header: data:image/png;base64
    mime = header.split(":", 1)[1].split(";", 1)[0]
    raw = base64.b64decode(b64)
    return mime, raw


class GeminiChatCompletionSampler:
    """
    Sample from Gemini using Google's official Gen AI SDK (google-genai).

    - Sync: client.models.generate_content(...)
    - JSON mode: response_mime_type="application/json"
    - Multimodal: supports OpenAI-like content parts:
        [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
    """

    def __init__(
            self,
            model: str = "gemini-2.5-pro",
            system_message: Optional[str] = None,
            temperature: float = 0.5,
            max_tokens: int = 1024,
            api_key: Optional[str] = None,
            vertexai: bool = False,
            project: Optional[str] = None,
            location: Optional[str] = None,
    ):
        # If api_key is None, SDK will read GOOGLE_API_KEY / GEMINI_API_KEY automatically. :contentReference[oaicite:4]{index=4}
        self.client = genai.Client(
            api_key=api_key,
            vertexai=vertexai if vertexai else None,
            project=project,
            location=location,
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _handle_text(self, text: str) -> types.Part:
        return types.Part.from_text(text=text)

    def _handle_image_base64(
            self,
            image_b64: str,
            mime_type: str = "image/png",
    ) -> types.Part:
        raw = base64.b64decode(image_b64)
        return types.Part.from_bytes(data=raw, mime_type=mime_type)

    def _handle_image_data_url(self, data_url: str) -> types.Part:
        mime, raw = _strip_data_url(data_url)
        return types.Part.from_bytes(data=raw, mime_type=mime)

    def _pack_contents(self, message_list: MessageList) -> List[types.Content]:
        """
        Convert OpenAI-like messages to Gemini contents (list[types.Content]).
        Roles: user / model. (We map assistant -> model, ignore system here.)
        """
        contents: List[types.Content] = []

        for msg in message_list:
            role = str(msg.get("role", "user"))
            if role == "assistant":
                role = "model"
            if role == "system":
                # system handled via config.system_instruction
                continue

            content = msg.get("content", "")
            parts: List[types.Part] = []

            # 1) Plain string
            if isinstance(content, str):
                parts.append(self._handle_text(content))

            # 2) OpenAI-style multimodal parts list
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        parts.append(self._handle_text(str(item)))
                        continue

                    t = item.get("type")
                    if t == "text":
                        parts.append(self._handle_text(item.get("text", "")))
                    elif t == "image_url":
                        url = (item.get("image_url") or {}).get("url", "")
                        if url.startswith("data:"):
                            parts.append(self._handle_image_data_url(url))
                        else:
                            # If you want to support http(s) images: you'd need to download bytes yourself
                            # or use Part.from_uri for supported URIs (e.g., GCS). Keeping it strict here.
                            raise ValueError(f"Unsupported image url (not data URL): {url[:50]}...")
                    else:
                        parts.append(self._handle_text(str(item)))

            else:
                parts.append(self._handle_text(str(content)))

            contents.append(types.Content(role=role, parts=parts))

        return contents

    def __call__(
            self,
            message_list: MessageList,
            temperature: Optional[float] = None,
            response_format: Optional[str] = None,  # 'normal' or 'json'
    ) -> Tuple[str, Any]:
        trial = 0
        while True:
            try:
                contents = self._pack_contents(message_list)

                config = types.GenerateContentConfig(
                    temperature=self.temperature if temperature is None else temperature,
                    max_output_tokens=self.max_tokens,
                )

                if self.system_message:
                    # System instruction via config is the intended pattern in the new SDK. :contentReference[oaicite:5]{index=5}
                    config.system_instruction = self.system_message

                if response_format != "normal":
                    # JSON response: set response_mime_type="application/json". :contentReference[oaicite:6]{index=6}
                    config.response_mime_type = "application/json"

                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                text = getattr(resp, "text", None)
                if text is None:
                    # Fallback: concatenate parts
                    text = "".join([p.text or "" for p in (resp.candidates[0].content.parts or [])])

                usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usage", None)
                return text, usage

            except Exception as e:
                exception_backoff = 2 ** trial
                print(f"Gemini exception; retry {trial} after {exception_backoff} sec:", e)
                time.sleep(exception_backoff)
                trial += 1
                if trial == 3:
                    print("Bad Request (or persistent error) after retries:", e)
                    return "", None


class AsyncGeminiChatCompletionSamplerAiohttp:
    """
    Async sampler for Gemini via google-genai using aiohttp transport.

    Requirements:
      pip install "google-genai[aiohttp]"

    Notes:
      - Pass aiohttp.ClientSession.request(...) kwargs via HttpOptions(async_client_args={...})
      - Close the async client (aclose) to avoid unclosed session warnings.
    """

    def __init__(
            self,
            model: str = "gemini-2.5-pro",
            system_message: Optional[str] = None,
            temperature: float = 0.5,
            max_tokens: int = 1024,
            response_format: str = "json",  # "json" or "normal"
            api_key: Optional[str] = None,
            # ---- aiohttp knobs ----
            timeout_s: float = 60.0,
            proxy: Optional[str] = None,
            ssl: Any = None,  # aiohttp SSL context or bool
            cookies: Optional[Dict[str, str]] = None,
            headers: Optional[Dict[str, str]] = None,
            # You can pass any other aiohttp request/session options via async_client_args_extra
            async_client_args_extra: Optional[Dict[str, Any]] = None,
            # ---- optional: Vertex AI backend ----
            vertexai: bool = False,
            project: Optional[str] = None,
            location: Optional[str] = None,
            api_version: Optional[str] = None,  # e.g. "v1" / "v1alpha"
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format

        # aiohttp.ClientSession.request kwargs are passed via async_client_args (official README). :contentReference[oaicite:2]{index=2}
        async_client_args: Dict[str, Any] = {
            # timeout in aiohttp can be a float (total seconds) or ClientTimeout; SDK forwards it
            "timeout": timeout_s,
        }
        if proxy is not None:
            async_client_args["proxy"] = proxy
        if ssl is not None:
            async_client_args["ssl"] = ssl
        if cookies is not None:
            async_client_args["cookies"] = cookies
        if headers is not None:
            async_client_args["headers"] = headers
        if async_client_args_extra:
            async_client_args.update(async_client_args_extra)

        http_options = types.HttpOptions(async_client_args=async_client_args)
        if api_version:
            http_options.api_version = api_version

        self._client = genai.Client(
            api_key=api_key,  # if None, SDK reads GOOGLE_API_KEY / GEMINI_API_KEY automatically :contentReference[oaicite:3]{index=3}
            vertexai=vertexai if vertexai else None,
            project=project,
            location=location,
            http_options=http_options,
        ).aio

    async def aclose(self):
        # Officially supported close method for async client. :contentReference[oaicite:4]{index=4}
        await self._client.aclose()

    def _to_contents(self, message_list: MessageList) -> List[types.Content]:
        contents: List[types.Content] = []
        for msg in message_list:
            role = str(msg.get("role", "user"))
            if role == "assistant":
                role = "model"
            if role == "system":
                continue  # system handled via config.system_instruction

            content = msg.get("content", "")
            parts: List[types.Part] = []

            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                # OpenAI-like multimodal parts
                for item in content:
                    if not isinstance(item, dict):
                        parts.append(types.Part.from_text(text=str(item)))
                        continue
                    t = item.get("type")
                    if t == "text":
                        parts.append(types.Part.from_text(text=item.get("text", "")))
                    elif t == "image_url":
                        url = (item.get("image_url") or {}).get("url", "")
                        if url.startswith("data:"):
                            mime, raw = _strip_data_url(url)
                            parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
                        else:
                            raise ValueError(f"Unsupported image url (not data URL): {url[:80]}...")
                    else:
                        parts.append(types.Part.from_text(text=str(item)))
            else:
                parts.append(types.Part.from_text(text=str(content)))

            contents.append(types.Content(role=role, parts=parts))

        return contents

    async def __call__(
            self,
            message_list: MessageList,
            temperature: Optional[float] = None,
            response_format: Optional[str] = None,  # "json" or "normal"
    ):
        trial = 0
        while True:
            try:
                contents = self._to_contents(message_list)

                config = types.GenerateContentConfig(
                    temperature=self.temperature if temperature is None else temperature,
                    max_output_tokens=self.max_tokens,
                )
                if self.system_message:
                    config.system_instruction = self.system_message

                eff_fmt = response_format or self.response_format
                if eff_fmt != "normal":
                    # JSON mode via response_mime_type is the intended mechanism. :contentReference[oaicite:5]{index=5}
                    config.response_mime_type = "application/json"

                resp = await self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                text = getattr(resp, "text", None)
                if text is None:
                    # fallback: join candidate parts
                    parts = resp.candidates[0].content.parts or []
                    text = "".join([p.text or "" for p in parts])

                usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usage", None)
                return text, usage

            except Exception as e:
                backoff = 2 ** trial
                print(f"Gemini(aiohttp) exception; retry {trial} after {backoff}s:", e)
                await asyncio.sleep(backoff)
                trial += 1
                if trial == 3:
                    print("Persistent error after retries:", e)
                    return "", None
