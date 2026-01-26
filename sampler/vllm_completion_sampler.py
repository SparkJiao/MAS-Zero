import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from time import process_time_ns
from typing import Any
import requests

import aiohttp

from utils import extract_xml
from .sampler_base import SamplerBase, MessageList


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            system_message: str | None = None,
            temperature: float = 0.5,
            model: str | None = None,
            max_tokens: int = 4096,
            response_format: str = "json"
    ):

        # model_api_map = {
        #     'qwen-2.5-32b-instr': '8082',
        #     'qwen3-30b-a3b': '8000',
        # }
        self.api_key_name = "API_KEY"
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv(self.api_key_name, "")

        url_base = os.getenv("BASE_URL", "http://localhost:8000/v1/chat/completions")
        self.url_base = url_base

        self.model = model

        self.response_format = response_format

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def xml_to_json(self, ori_answer):
        output_dict = OrderedDict()  # <-- keep insertion order
        tag_names = re.findall(r"</?(\w+)>", ori_answer)
        ordered_unique_tags = list(OrderedDict.fromkeys(tag_names))
        print('tag_names: ', tag_names)

        for tag in ordered_unique_tags:
            if all(t not in tag for t in ['A', 'B', 'C', 'D', 'sub', 'S_y', 'TOO_HARD', 'command', 'new', 'data', 'comment']):
                tag_text = extract_xml(ori_answer, tag)
                output_dict[tag] = tag_text
        json_string = json.dumps(output_dict, indent=4)
        return json_string

    def __call__(self, message_list: MessageList, temperature=None, response_format=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if not isinstance(message['content'], str):
                        message_list[message_id]['content'] = str(message['content'])

                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature if temperature is not None else self.temperature
                }

                # 发送同步请求
                response = requests.post(self.url_base, headers=headers, json=payload, timeout=7200)

                if response.status_code == 200:
                    response = response.json()
                else:
                    response.raise_for_status()
                # print('response: ', response, flush=True)
                ori_answer = response["choices"][0]["message"]["content"]
                # print('ori_answer: ',ori_answer)

                # json_string = self.xml_to_json(ori_answer)
                if self.response_format == "xml":
                    json_string = self.xml_to_json(ori_answer)
                else:
                    json_string = ori_answer

                return json_string, response["usage"]
            except Exception as e:
                import traceback
                traceback.print_exc()
                exception_backoff = 2 ** trial  # expontial back off
                print(
                    f"VLLM: Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                    flush=True
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception


class AsyncChatCompletionSampler(ChatCompletionSampler):
    async def __call__(self, message_list: MessageList, temperature=None, response_format=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0

        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if not isinstance(message['content'], str):
                        message_list[message_id]['content'] = str(message['content'])

                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature if temperature is not None else self.temperature
                }
                # 异步请求
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url_base, headers=headers, json=payload, timeout=7200) as response:
                        if response.status == 200:
                            result = await response.json()
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(
                                f"Request failed: HTTP {response.status}, {error_text}"
                            )

                # print('response: ',response)
                ori_answer = result["choices"][0]["message"]["content"]
                # print('ori_answer: ',ori_answer)
                if self.response_format == "xml":
                    json_string = self.xml_to_json(ori_answer)
                else:
                    json_string = ori_answer

                # json_string = ori_answer

                # print(json_string)
                return json_string, result["usage"]
            except Exception as e:
                import traceback
                traceback.print_exc()
                exception_backoff = 2 ** trial  # exponential back off
                print(
                    f"VLLM: Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial > 1:
                    return {}, {}
            # unknown error shall throw exception


if __name__ == '__main__':
    client = AsyncChatCompletionSampler(model="gpt-oss-120b", response_format="json")

    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Output a json string with single field: `capital`."},
    ]

    results = asyncio.run(client(history))
    print(results)
