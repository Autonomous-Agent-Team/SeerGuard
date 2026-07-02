import os
import re
import cv2
import time
from PIL import Image
from openai import OpenAI
from datetime import datetime

from mobile_safety.agent.LLM_agent import LLM_Agent
from mobile_safety.agent.utils import encode_image, parse_response

# Use .get() to avoid crashes when environment variables are not set, provide default path as fallback
_WORK_PATH = os.environ.get("MOBILE_SAFETY_HOME", ".")

class GeminiAgent(LLM_Agent):
    def __init__(
        self, model_name="gemini-3.1-pro-preview", seed=42, port=5554, safety_settings=None
    ):
        super().__init__(model_name, seed, port)

        self.model = OpenAI(
            base_url=os.environ.get("GEMINI_API_BASE", "YOUR_API_URL_HERE"),
            api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE"),
            timeout=120.0  # 120 second timeout
        )

        # Set safety settings to lowest (BLOCK_NONE) to prevent Safety Bench tasks from being blocked
        self.safety_settings = safety_settings or [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]

    def get_response(self, timestep=None, system_prompt=None, user_prompt=None):
        img_pil_path = self.save_image(timestep)

        # add context to user prompt
        current_user_prompt = re.sub(r"<context>", self.context, user_prompt)

        start_time = datetime.now()

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": current_user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(img_pil_path)}"
                        },
                    },
                ],
            },
        ]

        max_retries = 3
        response_dict = {"action": None, "context": ""}
        raw_response_text = ""

        # Introduce retry mechanism
        for attempt in range(max_retries):
            try:
                # Compatible with cases where config attribute or keys might not exist in LLM_Agent
                temp = getattr(self, "config", {}).get("temperature", 0.7)
                max_tok = getattr(self, "config", {}).get("max_tokens", 1024)
                top_p_val = getattr(self, "config", {}).get("top_p", 1.0)

                # Create completion
                completion = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    top_p=top_p_val,
                    extra_body={"safety_settings": self.safety_settings} # Try to pass safety parameters to relay
                )

                raw_response_text = completion.choices[0].message.content
                print(f"\n[DEBUG Attempt {attempt+1}/{max_retries}] Raw model output:\n{raw_response_text}\n")

                # Handle case where model might return empty value due to safety reasons
                if not raw_response_text:
                    finish_reason = completion.choices[0].finish_reason
                    print(f"[WARNING] Model returned empty. Finish reason: {finish_reason}")
                    continue

                # parse response
                response_dict = parse_response(raw_response_text)

                # If parsed action is still None, it means model output nonsense, continue retry
                if response_dict.get("action") is None:
                    print(f"[WARNING Attempt {attempt+1}] Parsed action is None. Retrying...")
                    continue

                # Successfully got valid action, update context and break loop
                if response_dict.get("context", "") != "":
                    self.context = response_dict["context"]

                response_dict["raw_response"] = raw_response_text
                break

            except Exception as e:
                print(f"[ERROR Attempt {attempt+1}] Exception during API call or parsing: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2) # Pause briefly after failure before retry, to prevent triggering rate limits

        # If all retries failed
        if response_dict.get("action") is None:
            print("⚠️ Maximum retries reached, unable to get valid response.")
            response_dict["raw_response"] = str(raw_response_text)

        end_time = datetime.now()
        time_elapsed = end_time - start_time

        return response_dict, time_elapsed

    def save_image(self, timestep=None):
        img_obs = timestep.curr_obs["pixel"]
        img_cv = cv2.resize(img_obs, dsize=(1024, 2048), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_cv)
        img_pil_path = f"{_WORK_PATH}/logs/tmp_{self.port}.png"

        # Add directory check to prevent errors if logs folder doesn't exist
        os.makedirs(os.path.dirname(img_pil_path), exist_ok=True)

        img_pil.save(img_pil_path)

        return img_pil_path
