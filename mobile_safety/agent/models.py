# models.py
import base64
import logging
import requests
from typing import Any, Dict, List, Optional, Union
import io
import os

from PIL import Image

logger = logging.getLogger(__name__)


MODEL_DICT = {
    "qwen3-vl-8b-instruct": {
        "model_name": "qwen3-vl-8b-instruct",
        "api_url": "YOUR_API_URL_HERE",
        "api_key": "YOUR_API_KEY_HERE"
    },
    "gpt-5.1": {
        "model_name": "gpt-5.1",
        "api_url": "YOUR_API_URL_HERE",
        "api_key": "YOUR_API_KEY_HERE"
    },
    "gemini-3.1-pro-preview": {
        "model_name": "gemini-3.1-pro-preview",
        "api_url": "YOUR_API_URL_HERE",
        "api_key": "YOUR_API_KEY_HERE"
    },
    "SeerGuard": {
        "model_name": "SeerGuard",
        "api_url": "YOUR_API_URL_HERE", 
        "api_key": "YOUR_API_KEY_HERE"
    }
}



class ModelWrapper:
    """
    Generic wrapper for any model exposing an OpenAI-compatible API:
    - OpenAI
    - Qwen/Qwen-VL official API
    - DeepSeek
    - vLLM OpenAI server: http://host/v1/chat/completions

    Supports:
    - LLM text-only
    - VLM with multi-image input
    """

    def __init__(
        self,
        model_name: str,
        timeout: int = 300,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
            model_name: name of model or deployment
            api_url: full URL of /v1/chat/completions
            api_key: optional; some vLLM deployments do not require keys
            extra_params: default params merged into each request (optional)
        """
        self.model_name = MODEL_DICT[model_name]['model_name']
        self.api_url = MODEL_DICT[model_name]['api_url']
        self.api_key = MODEL_DICT[model_name]['api_key']
        self.timeout = timeout
        self.extra_params = extra_params or {}

    # ------------------------------------------------------------------
    # Encode images to base64
    # ------------------------------------------------------------------
    def _encode_images(self, images: Optional[List[Union[str, bytes, Image.Image]]]):
        """
        Encode images into base64 dicts suitable for model input.

        Args:
            images: list of
                - bytes: raw image bytes
                - str: file path to an image
                - PIL.Image.Image: PIL Image object

        Returns:
            List[Dict]: [{"type": "input_image", "image": base64_str}, ...]
        """
        if not images:
            return []

        encoded_images = []

        for img in images:
            img_bytes = None

            if isinstance(img, bytes):
                img_bytes = img
            elif isinstance(img, str):
                if not os.path.isfile(img):
                    raise FileNotFoundError(f"Image path not found: {img}")
                with open(img, "rb") as f:
                    img_bytes = f.read()
            elif isinstance(img, Image.Image):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            encoded_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"}
            })

        return encoded_images

    # ------------------------------------------------------------------
    # Main chat call
    # ------------------------------------------------------------------
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[bytes]] = None,
        **kwargs,
    ) -> str:
        """
        Core API:
            resp_text = wrapper.chat(system, user, images=[...])

        Returns: raw model output string
        """
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
            ]
        else:
            messages = []

        # If VLM → add images before user text
        if images:
            messages.append({
                "role": "user",
                "content": self._encode_images(images) + [
                    {"type": "text", "text": user_prompt}
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": user_prompt,
            })

        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
        }

        # Add extra_params as default config
        body.update(self.extra_params)

        # User override: chat(..., temperature=0.1)
        body.update(kwargs)

        logger.debug(f"[ModelWrapper] Request body: {body}")

        # HTTP headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"  # optional

        # HTTP request
        resp = requests.post(
            self.api_url,
            json=body,
            headers=headers,
            timeout=self.timeout
        )

        if resp.status_code != 200:
            logger.error(f"Model API Error {resp.status_code}: {resp.text}")
            raise RuntimeError(f"Model API Error: {resp.text}")

        data = resp.json()

        usage_final = data.get("usage", {})
        prompt_tokens = usage_final.get("prompt_tokens", 0)
        completion_tokens = usage_final.get("completion_tokens", 0)

        logger.debug(f"[ModelWrapper] Raw response: {data}")

        # Standard OpenAI response format
        return data["choices"][0]["message"]["content"], prompt_tokens, completion_tokens
