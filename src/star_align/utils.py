import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeVar

import openai
import tenacity
import tiktoken

N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2


def read_jsonl(path: str | Path) -> list[Any]:
    """Read lines of JSON from a file (including '\n')."""
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, data: Sequence[Mapping], mode: str = "w"):
    # cannot use `dict` here as it is invariant
    with Path(path).open(mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def retry(errors: Any, max_attempts: int = 5):
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(errors),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=20),
        stop=tenacity.stop_after_attempt(max_attempts),
        before_sleep=print,
    )


ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncClient()

    @retry(ERRORS)
    def chat_completions_with_backoff(self, *args, **kwargs):
        return self.client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    def completions_with_backoff(self, *args, **kwargs):
        return self.client.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def chat_completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.completions.create(*args, **kwargs)

    async def delayed_request(
        self,
        request: dict[str, Any],
        mode: Literal["chat", "completion"],
        delay: float | None,
    ):
        """Prevent quantized rate limit:
        https://help.openai.com/en/articles/6891753-rate-limit-advice"""
        if delay is not None:
            # synchronized sleep
            time.sleep(delay)
        if mode == "chat":
            func = self.chat_completions_with_backoff_async
        else:
            func = self.completions_with_backoff_async
        return await func(**request)

    async def dispatch_chat_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch chat completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [self.delayed_request(request, "chat", delay) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def dispatch_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [
            self.delayed_request(request, "completion", delay) for request in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    # encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_fingerprint(*args: Any, hash_length: int | None = None) -> str:
    combined = "".join(map(str, args))
    content = hashlib.sha256(combined.encode()).hexdigest()
    if hash_length is not None:
        content = content[:hash_length]
    return content


def find_code_blocks(response: str, tag: str | None = None) -> list[str]:
    """Find all enclosed code blocks in the response, optionally filtering by language tag."""
    all_indices = find_codeblock_indices(response, tag)
    return [response[start:end].strip() for start, end in all_indices]


def find_codeblock_indices(
    response: str, tag: str | None = None
) -> list[tuple[int, int]]:
    """Find all enclosed code blocks in the response, optionally filtering by language tag."""
    all_indices: list[tuple[int, int]] = []
    search_start = (
        0  # Variable to keep track of where to start searching for the next code block
    )

    while "```" in response[search_start:]:
        # Find the start of the code block (excluding the backticks)
        code_start_index = response.find("```", search_start) + 3

        # Find the end of the language tag line (or the start of the code if no tag line)
        code_start_endline = response.find("\n", code_start_index)
        if code_start_endline == -1:  # Handle case where there's no newline after ```
            code_start_endline = code_start_index

        # Extract the language tag (if any)
        extracted_tag = response[code_start_index:code_start_endline].strip()

        # Adjust the start index if a language tag is found
        if extracted_tag:
            actual_code_start = code_start_endline + 1
        else:
            actual_code_start = code_start_index

        # Find the end of the code block
        code_end_index = response.find("```", actual_code_start)
        if code_end_index == -1:
            break  # Exit if there's no closing ```

        # Extract the code
        # code = response[actual_code_start:code_end_index].strip()

        # Check if the extracted code block matches the requested language tag (if any)
        if tag is None or extracted_tag.lower() == tag.lower():
            all_indices.append((actual_code_start, code_end_index))

        # Update the search_start to look for the next code block
        search_start = code_end_index + 3

    return all_indices


DEFAULT_TEMPLATE = """\
### Instruction
{instruction}

### Response
{response}"""


def is_base_model(tokenizer_name: str) -> bool:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer.chat_template is None and "octocoder" not in tokenizer_name


OCTOCODER_CHAT_TEMPLATE = """\
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{ raise_exception('System messages are not allowed in this template.') }}
    {%- else %}
        {%- if message['role'] == 'user' %}
{{'Question: ' + message['content'] + '\n\n'}}
        {%- else %}
{{'Answer: ' + message['content'] + '\n\n'}}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{{'Question: '}}"""


def infer_prompt_template(tokenizer_name: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if "octocoder" in tokenizer_name:
        tokenizer.chat_template = OCTOCODER_CHAT_TEMPLATE
    if tokenizer.chat_template is not None:
        template = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "{instruction}"},
                {"role": "assistant", "content": "{response}"},
            ],
            tokenize=False,
        )
    else:
        template = DEFAULT_TEMPLATE
    end_index = template.rindex("{response}") + len("{response}")
    template = template[:end_index]
    return template
