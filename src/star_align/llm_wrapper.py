import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal

import torch
from transformers import AutoModelForCausalLM, AutoModelWithLMHead, AutoTokenizer
from transformers import GenerationConfig as TransformersGenerationConfig
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# from peft import PeftModel, PeftConfig

# Tokenization side modeling
PaddingSide = Literal["left", "right"]
# Input: a batch of chat pieces; Output: a batch of instructions and responses
# The instances should encode in a way that the model can predict response from instruction
InputIds = list[int]

# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


@dataclass(frozen=True)
class DecodingConfig:
    skip_special_tokens: bool

    @staticmethod
    def default() -> "DecodingConfig":
        return DecodingConfig(skip_special_tokens=True)


# TransformChatPieceFunc = Callable[[ChatPiece], tuple[str, str]]


@dataclass(frozen=True)
class EncodingConfig:
    add_bos: bool
    add_eos: bool
    truncation: int | None = field(default=None)

    @staticmethod
    def default() -> "EncodingConfig":
        return EncodingConfig(add_bos=False, add_eos=False)


@dataclass(frozen=True)
class TokenizationContext:
    tokenizer: PreTrainedTokenizer
    pad_token_id: int
    bos_token: str
    eos_token: str

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @staticmethod
    def from_model_key(
        model_key: str, model_name_or_path: str | None = None
    ) -> "TokenizationContext":
        # use_fast = model_key not in SupportedModelKeys.codellama_models()
        use_fast = True
        # if model_name_or_path is None:
        #     model_name_or_path = model_key
        # TODO: check if tokenizers cannot be loaded with path
        model_name_or_path = model_key
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
        tokenization_context = TokenizationContext.from_tokenizer(tokenizer)
        return tokenization_context

    @staticmethod
    def from_tokenizer(tokenizer: PreTrainedTokenizer) -> "TokenizationContext":
        if (pad_token_id := tokenizer.pad_token_id) is None:
            pad_token_id = tokenizer.eos_token_id
        assert pad_token_id is not None
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        return TokenizationContext(
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    def encode(self, config: EncodingConfig, text_list: list[str]) -> list[list[int]]:
        # eos_token = self.eos_token if config.add_eos else ""
        # bos_token = self.bos_token if config.add_bos else ""
        # if eos_token != "" or bos_token != "":
        #     text_list = [f"{bos_token}{text}{eos_token}" for text in text_list]
        # The string concatenation above may not always work for all tokenizers (strange).
        # e.g., when codellama's tokenizer is used with "<s>[INST]".
        if config.truncation is not None:
            extra_args = dict(truncation=True, max_length=config.truncation)
        else:
            extra_args = {}
        input_ids = self.tokenizer(
            text_list,
            add_special_tokens=False,
            **extra_args,
        )["input_ids"]
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        bos_token_ids = (
            [bos_token_id] if config.add_bos and bos_token_id is not None else []
        )
        eos_token_ids = (
            [eos_token_id] if config.add_eos and eos_token_id is not None else []
        )
        if len(bos_token_ids) > 0 or len(eos_token_ids) > 0:
            input_ids = [
                bos_token_ids + input_id + eos_token_ids for input_id in input_ids
            ]
        return input_ids

    def decode(
        self, config: DecodingConfig, input_ids: list[InputIds] | torch.Tensor
    ) -> list[str]:
        return self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=config.skip_special_tokens
        )

    def encode_with_padding(
        self, padding_side: PaddingSide, config: EncodingConfig, text_list: list[str]
    ) -> torch.Tensor:
        input_ids_unpadded = self.encode(config, text_list)
        return pad_sequences(
            sequences=input_ids_unpadded,
            pad_value=self.pad_token_id,
            padding_side=padding_side,
        )


def pad_sequences(
    sequences: list[list[int]],
    pad_value: int,
    padding_side: Literal["left", "right"],
    dtype: torch.dtype = torch.long,
    padding_length: int | None = None,
) -> torch.Tensor:
    tensors = [torch.tensor(sequence, dtype=dtype) for sequence in sequences]
    max_len = max(len(sequence) for sequence in sequences)
    if padding_length is not None:
        assert padding_length >= max_len, "padding_length must be >= max_len"
        max_len = padding_length
    if padding_side == "right":
        result = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_value
        )
        remaining_length = max_len - result.shape[-1]
        # padding matrix of (batch_size * remaining_length)
        shape = result.shape[:-1] + (remaining_length,)
        padding_matrix = torch.full(shape, pad_value, dtype=dtype)
        result = torch.cat([result, padding_matrix], dim=-1)
    else:
        padded_tensors: list[torch.Tensor] = []
        for tensor in tensors:
            n_pad_values = max_len - len(tensor)
            padded_values = torch.full((n_pad_values,), pad_value, dtype=dtype)
            padded_tensor = torch.cat([padded_values, tensor], dim=0)
            assert len(padded_tensor) == max_len
            padded_tensors.append(padded_tensor)
        result = torch.stack(padded_tensors, dim=0)
    assert result.shape == torch.Size([len(sequences), max_len])
    return result


# Inference side modeling
@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    top_p: float
    temperature: float
    max_length: int = field(
        default=99999999999999999,
        metadata={
            "help": "The max length of the sequence to generate, including inputs."
            "Will be considered in tandem with max_new_tokens. Whichever is more restrictive will be used."
        },
    )

    def to_transformers_generation_config(
        self, eos_token_id: int, pad_token_id: int
    ) -> TransformersGenerationConfig:
        do_sample = self.temperature != 0.0
        kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            kwargs["temperature"] = self.temperature
        return TransformersGenerationConfig(**kwargs)

    def with_max_new_tokens_being(self, max_new_tokens: int) -> "GenerationConfig":
        return GenerationConfig(max_new_tokens, self.top_p, self.temperature)

    @staticmethod
    def default() -> "GenerationConfig":
        return GenerationConfig(200, 1.0, 1.0)


@dataclass(frozen=True)
class Response:
    raw_inputs: torch.Tensor
    raw_outputs: torch.Tensor
    decoded_outputs: list[str]


@dataclass
class ModelContext:
    tokenization_context: TokenizationContext
    model: PreTrainedModel
    max_context_size: int

    def generate(
        self,
        config: GenerationConfig,
        input_ids: torch.Tensor,
        stop_tokens: list[str] | None = None,
    ) -> torch.Tensor:
        """Raise ValueError when input_ids exceeds the context."""
        # NOTE: this implementation is only for decoder-only models
        # Recalculate the max number of tokens to avoid overflowing the context window
        input_len = input_ids.shape[1]
        if input_len >= self.max_context_size:
            raise ValueError(
                f"Input length {input_len} >= Context size {self.max_context_size}"
            )
        if input_len >= config.max_length:
            raise ValueError(
                f"Input length {input_len} >= Max length {config.max_length}"
            )
        assert input_len < self.max_context_size
        assert input_len < config.max_length

        max_new_tokens = min(
            self.max_context_size - input_len,
            config.max_new_tokens,
            config.max_length - input_len,
        )
        config = config.with_max_new_tokens_being(max_new_tokens)

        tf_config = config.to_transformers_generation_config(
            eos_token_id=self.tokenization_context.eos_token_id,
            pad_token_id=self.tokenization_context.pad_token_id,
        )
        attention_mask = input_ids.ne(self.tokenization_context.pad_token_id)
        # breakpoint()
        extra_kwargs: dict = {}
        if stop_tokens is not None:
            stopping_criteria = StoppingCriteriaList(
                [
                    EndOfFunctionCriteria(
                        start_length=len(input_ids[0]),
                        eos=stop_tokens,
                        tokenizer=self.tokenization_context.tokenizer,
                    )
                ]
            )
            extra_kwargs["stopping_criteria"] = stopping_criteria
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=tf_config,
            **extra_kwargs,
        )
        # input_len = input_ids.shape[1]
        return outputs[:, input_len:]

    def complete(
        self,
        config: GenerationConfig,
        prompts: list[str],
        stop_tokens: list[str] | None = None,
    ) -> Response:
        encoding_config = EncodingConfig(add_bos=True, add_eos=False)
        input_ids = self.tokenization_context.encode_with_padding(
            "left", encoding_config, prompts
        )
        input_ids = input_ids.to(self.model.device)
        output_ids = self.generate(config, input_ids, stop_tokens)
        decoding_config = DecodingConfig(skip_special_tokens=True)
        output_strings = self.tokenization_context.decode(decoding_config, output_ids)
        return Response(
            raw_inputs=input_ids,
            raw_outputs=output_ids,
            decoded_outputs=output_strings,
        )

class SupportedModelKeys(Enum):
    # StarCoder-based models
    STARCODER_15B = "bigcode/starcoder"
    WIZARDCODER_STARCODER_15B = "WizardLM/WizardCoder-15B-V1.0"

    # CodeLlama-based models
    WIZARDCODER_CODELLAMA_PYTHON_7B = "WizardLM/WizardCoder-Python-7B-V1.0"
    WIZARDCODER_CODELLAMA_PYTHON_13B = "WizardLM/WizardCoder-Python-13B-V1.0"
    WIZARDCODER_CODELLAMA_PYTHON_34B = "WizardLM/WizardCoder-Python-34B-V1.0"
    CODELLAMA_PYTHON_7B = "codellama/CodeLlama-7b-Python-hf"
    CODELLAMA_PYTHON_13B = "codellama/CodeLlama-13b-Python-hf"
    CODELLAMA_PYTHON_34B = "codellama/CodeLlama-34b-Python-hf"

    # DeepSeek-Coder-based models
    DEEPSEEK_CODER_1_3B = "deepseek-ai/deepseek-coder-1.3b-base"
    DEEPSEEK_CODER_6_7B = "deepseek-ai/deepseek-coder-6.7b-base"
    DEEPSEEK_CODER_33B = "deepseek-ai/deepseek-coder-33b-base"

    @staticmethod
    def all() -> list[str]:
        return [member.value for member in SupportedModelKeys]

    @staticmethod
    def codellama_models() -> list[str]:
        return [
            SupportedModelKeys.CODELLAMA_PYTHON_7B.value,
            SupportedModelKeys.CODELLAMA_PYTHON_13B.value,
            SupportedModelKeys.CODELLAMA_PYTHON_34B.value,
            # SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_7B.value,
            # SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_13B.value,
            # SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_34B.value,
        ]

    @staticmethod
    def codellama_based_models() -> list[str]:
        return SupportedModelKeys.codellama_models() + [
            SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_7B.value,
            SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_13B.value,
            SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_34B.value,
        ]

    @staticmethod
    def starcoder_based_models() -> list[str]:
        return [
            SupportedModelKeys.STARCODER_15B.value,
            SupportedModelKeys.WIZARDCODER_STARCODER_15B.value,
        ]

    @staticmethod
    def deepseekcoder_based_models() -> list[str]:
        return [
            SupportedModelKeys.DEEPSEEK_CODER_1_3B.value,
            SupportedModelKeys.DEEPSEEK_CODER_6_7B.value,
            SupportedModelKeys.DEEPSEEK_CODER_33B.value,
        ]


def get_model_context(
    model_key: str,
    model_name_or_path: str | None = None,
    tokenization_context: TokenizationContext | None = None,
    inference_mode: bool = True,
    use_flash_attention: bool = False,
    attention_dropout: float | None = None,
    residual_dropout: float | None = None,
    embedding_dropout: float | None = None,
) -> ModelContext:
    # `model_key` defines the model and the tokenizer to use, while `model_name_or_path`
    # defines where to load the weights. It can be from a local directory.
    # assert model_key in SupportedModelKeys.all(), model_key
    if model_key not in SupportedModelKeys.all():
        import warnings

        warnings.warn(
            f"{model_key} not explicitly supported. This may or may not lead to unexpected behaviors."
        )
    if model_name_or_path is None:
        model_name_or_path = model_key
    if model_key in SupportedModelKeys.codellama_based_models():
        max_context_size = 16384
    elif model_key in SupportedModelKeys.starcoder_based_models():
        max_context_size = 8192
    elif model_key in SupportedModelKeys.deepseekcoder_based_models():
        max_context_size = 16384
    else:
        import warnings

        warnings.warn(
            f"{model_key} does not have a specified max context, using default 4096"
        )
        max_context_size = 4096
    if tokenization_context is None:
        tokenization_context = TokenizationContext.from_model_key(model_key)
    # TODO: check if all these models use bfloat16
    dtype = torch.bfloat16
    other_kwargs: dict = {}
    if inference_mode:
        other_kwargs["device_map"] = "auto"
    if use_flash_attention:
        # if "starcoder2" in model_key:
        #     other_kwargs["attn_implementation"] = "flash_attention_2"
        # else:
        import transformers

        if transformers.__version__ <= "4.35.0":
            other_kwargs["use_flash_attention_2"] = True
        else:
            other_kwargs["attn_implementation"] = "flash_attention_2"
        # other_kwargs["use_flash_attention_2"] = True
    # cls = AutoModelWithLMHead if "starcoder2-3b" in model_key else AutoModelForCausalLM

    if "starcoder" in model_key.lower():
        print("Hack for starcoder")
        attention_dropout = attention_dropout or 0.0
        residual_dropout = residual_dropout or 0.0
        embedding_dropout = embedding_dropout or 0.0

    if attention_dropout is not None:
        other_kwargs["attention_dropout"] = attention_dropout
    if residual_dropout is not None:
        other_kwargs["residual_dropout"] = residual_dropout
    if embedding_dropout is not None:
        other_kwargs["embedding_dropout"] = embedding_dropout
    # if (dropout := os.getenv("ATTENTION_DROPOUT")) is not None:
    #     other_kwargs["attention_dropout"] = float(dropout)
    #     print(f"Using attention dropout: {dropout}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        # hack
        # revision=os.getenv("REVISION"),
        **other_kwargs,
    )
    print("Successfully loaded model.")
    print(model.config)
    return ModelContext(tokenization_context, model, max_context_size)


def form_starcoder_infill(prefix: str, suffix: str) -> str:
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    return prompt


def form_codellama_infill(prefix: str, suffix: str) -> str:
    # NOTE: not using <FILL_ME> because it's treated as a special token
    # but we pass `add_special_tokens=False` to the tokenizer
    return f"▁<PRE>{prefix}▁<SUF>{suffix}▁<MID>"


def form_deepseekcoder_infill(
    tokenizer: PreTrainedTokenizer, prefix: str, suffix: str
) -> str:
    def get_token(idx: int) -> str:
        return tokenizer.convert_ids_to_tokens([idx])[0]

    FIM_PREFIX = get_token(32016)
    FIM_MIDDLE = get_token(32015)
    FIM_SUFFIX = get_token(32017)
    assert "begin" in FIM_PREFIX and "hole" in FIM_MIDDLE and "end" in FIM_SUFFIX
    prompt = f"{FIM_PREFIX}{prefix}{FIM_MIDDLE}{suffix}{FIM_SUFFIX}"
    return prompt


def create_infilling_prompt(
    model_key: str,
    prefix: str,
    suffix: str,
    tokenizer: PreTrainedTokenizer | None = None,
) -> str:
    if model_key in SupportedModelKeys.starcoder_based_models():
        return form_starcoder_infill(prefix, suffix)
    elif (
        model_key in SupportedModelKeys.codellama_based_models()
        and not "python" in model_key.lower()
    ):
        return form_codellama_infill(prefix, suffix)
    elif model_key in SupportedModelKeys.deepseekcoder_based_models():
        assert tokenizer is not None
        return form_deepseekcoder_infill(tokenizer, prefix, suffix)

    # TODO: other models
    assert False, f"Unsupported model key: {model_key}"
