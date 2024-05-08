import os
from dataclasses import dataclass, field
from typing import cast

import torch
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments

from star_align.llm_wrapper import (
    DecodingConfig,
    EncodingConfig,
    TokenizationContext,
    get_model_context,
    pad_sequences,
)
from star_align.prompt_template import CHAT_TEMPLATE
from star_align.prompt_template import SC2_INSTRUCT_PROMPT as PROMPT_TEMPLATE
from star_align.utils import N_CORES


@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None
    attention_dropout: float | None = field(default=None)
    residual_dropout: float | None = field(default=None)
    embedding_dropout: float | None = field(default=None)


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100


def map_dataset(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    if args.prompt_completion_mode:
        prompts = examples["prompt"]
        completions = examples["completion"]
    else:
        instructions = examples["instruction"]
        responses = examples["response"]

        prompts = [
            PROMPT_TEMPLATE.format(instruction=instruction, response="")
            for instruction in instructions
        ]
        completions = responses

    assert len(prompts) == len(completions)
    prompt_config = EncodingConfig(add_bos=True, add_eos=False)
    completion_config = EncodingConfig(add_bos=False, add_eos=True)
    prompt_id_batches = context.encode(prompt_config, prompts)
    completion_id_batches = context.encode(completion_config, completions)
    # prompt_id_batches = context.tokenization_context.encode(prompt_config, prompts)
    # completion_id_batches = context.tokenization_context.encode(
    #     completion_config, completions
    # )
    assert len(prompt_id_batches) == len(completion_id_batches)
    untruncated_input_ids = [
        (instruction_ids + response_ids)
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    exceeding_length = [
        len(input_id) > args.max_training_seq_length
        for input_id in untruncated_input_ids
    ]
    input_ids = [
        input_id[: args.max_training_seq_length] for input_id in untruncated_input_ids
    ]
    # NOTE: no need to set EOF to IGNORED_INDEX as it is *implicitly* ignored inside
    # the model.forward that shifts the logits left by 1
    labels = [
        (list(map(lambda _: IGNORED_INDEX, instruction_ids)) + response_ids)[
            : args.max_training_seq_length
        ]
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    # `len` of each returned value must be the same, which is required by `tokenizer.map`
    # After `map`, they are treated as individual pieces of data, not as a batch.
    assert len(input_ids) == len(labels)
    for input_id_batch, label_batch in zip(input_ids, labels):
        assert len(input_id_batch) == len(label_batch)
    print(context.decode(DecodingConfig.default(), input_ids[0:])[0])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "exceeding_length": exceeding_length,
    }


def get_data_collator(model_key: str, args: "Args", pad_token_id: int):
    """Pad input_ids to the right, create labels by setting the padding tokens to -100, and
    create attention_mask to ignore the padding tokens"""

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        padding_length = (
            args.max_training_seq_length if args.pad_to_max_length else None
        )
        input_ids = pad_sequences(
            input_ids_unpadded, pad_token_id, "right", padding_length=padding_length
        )
        labels = pad_sequences(
            labels_unpadded, IGNORED_INDEX, "right", padding_length=padding_length
        )

        assert input_ids.shape == labels.shape
        assert len(input_ids) == len(examples)
        # Enforced in `map_raw_dataset`
        assert input_ids.shape[-1] <= args.max_training_seq_length
        if args.pad_to_max_length:
            assert input_ids.shape[-1] == args.max_training_seq_length

        if "starcoder2" in model_key:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool)
        else:
            attention_mask = input_ids.ne(pad_token_id)
            # when bos == eos, the first token will be masked by mistake
            attention_mask[:, 0] = True
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate


@dataclass(frozen=True)
class Args:
    datafile_paths: list[str] = field(default_factory=list)
    max_training_seq_length: int = field(default=1216)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )
    use_flash_attention: bool = field(default=False)
    prompt_completion_mode: bool = field(default=False)


def train():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, Args))
    model_args, training_args, args = cast(
        tuple[ModelArguments, TrainingArguments, Args],
        parser.parse_args_into_dataclasses(),
    )
    dataset = load_dataset("json", data_files=args.datafile_paths, split="train")

    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key

    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    # if dataset_config.dpo_jsonl_path is None or dataset_config.dpo_sft:
    train_dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args, context=tokenization_context),
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,  # not args.overwrite_cache
        desc="Running tokenizer on train dataset",
    )
    msg = f"#Examples truncated: {sum(train_dataset['exceeding_length'])} / {len(train_dataset)}"
    print(msg)
    # else:
    #     train_dataset = dataset

    # Shuffling
    if training_args.eval_steps is None and training_args.evaluation_strategy == "no":
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        eval_dataset = None
    else:
        print("Splitting dataset")
        split_dataset = train_dataset.train_test_split(
            test_size=args.eval_dataset_size,
            shuffle=True,
            seed=training_args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    state = get_model_context(
        model_key,
        model_name_or_path,
        tokenization_context,
        inference_mode=False,
        use_flash_attention=args.use_flash_attention,
        attention_dropout=model_args.attention_dropout,
        residual_dropout=model_args.residual_dropout,
        embedding_dropout=model_args.embedding_dropout,
    )
    if "codeqwen" in model_key.lower():
        print(f"Hack for {model_key}")
        state.model.generation_config.do_sample = True

    print("Parallel mode:", training_args.parallel_mode)
    data_collator = get_data_collator(
        model_args.model_key, args, state.tokenization_context.pad_token_id
    )

    # neftune_noise_alpha
    trainer = Trainer(
        model=state.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )

    # NOTE: the checkpoint will override the initialized model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    state.tokenization_context.tokenizer.chat_template = CHAT_TEMPLATE
    state.tokenization_context.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
