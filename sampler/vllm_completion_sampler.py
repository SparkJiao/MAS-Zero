import aiohttp
import asyncio
import base64
import json
import openai
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from openai import OpenAI
from together import Together
from typing import Any
from typing import Any
from typing import Union, Optional, List, Dict

from utils import extract_xml

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError


OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
        "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
        + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


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
    ):

        model_api_map = {
            'qwen-2.5-32b-instr': '8082',
            'qwen3-30b-a3b': '8000',
        }
        self.api_key_name = "OPENAI_API_KEY"
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai_api_key = "EMPTY"

        base_port = os.getenv("BASE_PORT")
        if base_port:
            url_base = f"http://localhost:{base_port}/v1/chat/completions"
        else:
            url_base = f"http://localhost:{model_api_map[model]}/v1/chat/completions"
        self.url_base = url_base

        self.model = model

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

    def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if not isinstance(message['content'], str):
                        message_list[message_id]['content'] = str(message['content'])

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer None"
                }
                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature
                }

                # 发送同步请求
                response = requests.post(api_base, headers=headers, json=payload)

                if response.status_code == 200:
                    return response.json()
                else:
                    response.raise_for_status()
                # print('response: ',response)
                ori_answer = response.choices[0].message.content
                # print('ori_answer: ',ori_answer)

                json_string = self.xml_to_json(ori_answer)

                return json_string, response.usage
            except Exception as e:
                import traceback
                traceback.print_exc()
                exception_backoff = 2 ** trial  # expontial back off
                print(
                    f"VLLM: Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception


class AsyncChatCompletionSampler(ChatCompletionSampler):
    async def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer None"
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
                    "temperature": temperature
                }

                # 异步请求
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url_base, headers=headers, json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(
                                f"Request failed: HTTP {response.status}, {error_text}"
                            )

                # print('response: ',response)
                ori_answer = response.choices[0].message.content
                # print('ori_answer: ',ori_answer)

                json_string = self.xml_to_json(ori_answer)
                # json_string = ori_answer

                return json_string, response.usage
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
            # unknown error shall throw exception


if __name__ == '__main__':
    client = AsyncChatCompletionSampler(model="qwen3-30b-a3b")

    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    results = asyncio.run(client(history))
    print(results)
