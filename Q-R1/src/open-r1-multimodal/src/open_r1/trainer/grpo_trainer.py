# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
import copy
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, is_deepspeed_available
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from trl import GRPOTrainer
import re
from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
      #  print(self.mini_repeat_count, self.repeat_count, indexes)
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        self.with_revision = kwargs.get("with_revision", False)
        self.with_revision_omni = kwargs.get("with_revision_omni", False)
        self.logical_withllm = kwargs.get("logical_withllm", False)
        self.only_deterministic = kwargs.get("only_deterministic", False)
        self.load_from_local = kwargs.get("load_from_local", False)
        self.llm_to_instruct= kwargs.get("llm_to_instruct", False)
        self.use_drgrpo = kwargs.get("use_drgrpo", False)

        if self.with_revision:
            reward_funcs_revision = [vlm_module.iou_reward]
        if self.logical_withllm:
            reward_funcs_logical = [vlm_module.logical_reward]
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # model_init_kwargs["enable_audio_output"] = False
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        #     # Disable caching if gradient checkpointing is enabled (not supported)
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs, self.load_from_local)

        model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        if not self.load_from_local:
            model = model.thinker # for qwen-omni

        self.embedding_layer = model.get_input_embeddings()

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)


        def init_llm(model_name):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_init_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer

        if self.with_revision or self.logical_withllm or self.llm_to_instruct:
            self.revision_model, self.revision_tokenizer = init_llm('./code/R1-V-Qwen/Q-R1/Qwen2.5-0.5B-Instruct')


        # Reference model
        if is_deepspeed_zero3_enabled() or is_deepspeed_available():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
            if not self.load_from_local:
                self.ref_model = self.ref_model.thinker # for qwen-omni
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
            if not self.load_from_local:
                self.ref_model = self.ref_model.thinker # for qwen-omni
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id
        # print(processing_class.tokenizer)
        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        self.func_ids = {}
        total_trainable_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(f'train param: {name}')
                total_trainable_params += p.numel()
        

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if self.with_revision:
            if not isinstance(reward_funcs_revision, list):
                reward_funcs_revision = [reward_funcs_revision]
            for i, reward_func in enumerate(reward_funcs_revision):
                if isinstance(reward_func, str):
                    reward_funcs_revision[i] = AutoModelForSequenceClassification.from_pretrained(
                        reward_func, num_labels=1, **model_init_kwargs
                    )
            self.reward_funcs_revision = reward_funcs_revision
        
        if self.logical_withllm:
            if not isinstance(reward_funcs_logical, list):
                reward_funcs_logical = [reward_funcs_logical]
            for i, reward_func in enumerate(reward_funcs_logical):
                if isinstance(reward_func, str):
                    reward_funcs_logical[i] = AutoModelForSequenceClassification.from_pretrained(
                        reward_func, num_labels=1, **model_init_kwargs
                    )
            self.reward_funcs_logical = reward_funcs_logical

        if self.with_revision or self.logical_withllm:
            for i in range(len(self.reward_funcs)):
                self.func_ids[i] = 'origin'

        if self.with_revision:  
            self.reward_funcs = self.reward_funcs + self.reward_funcs_revision
            self.func_ids[len(self.reward_funcs) - 1] = 'revision'

        if self.logical_withllm:
            self.reward_funcs = self.reward_funcs + self.reward_funcs_logical
            self.func_ids[len(self.reward_funcs) - 1] = 'logical'


        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")


        self.num_gen_gt = args.num_gen_gt

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
        )

        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))
        self.beta = args.beta
        self.epsilon = args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
                if self.with_revision or self.logical_withllm or self.llm_to_instruct: 
                    self.revision_model = prepare_deepspeed(self.revision_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        if self.with_revision:
            for i, reward_func in enumerate(self.reward_funcs_revision):
                if isinstance(reward_func, PreTrainedModel):
                    self.reward_funcs_revision[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _revision_reward(self, completions, solution, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, sol in zip(contents, solution):
            question_answer_pair = self.generate_valid_json(content)

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model, revision=False) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        generation_config_ = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=self.processing_class.pad_token_id,
        )

        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            generation_config_.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))

        if self.only_deterministic:
            if device == torch.device('cuda:0'):
                generation_config_ = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    do_sample=False,  
                    pad_token_id=self.processing_class.pad_token_id,
                )

                if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
                    generation_config_.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
                    print(222, self.vlm_module.get_eos_token_id(processing_class))
            else:
                generation_config_ = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,  
                    temperature=1,
                    pad_token_id=self.processing_class.pad_token_id,
                )

                if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
                    generation_config_.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
                    print(222, self.vlm_module.get_eos_token_id(processing_class))


       # print(device)
        if self.num_gen_gt == 1:
            if device == torch.device('cuda:0'):
                for x in inputs:
                    x['prompt'][0]['content'][2]['text'] = x['prompt'][0]['content'][2]['text'].format(ground_truth_emotion=x['solution'])
            else:
                for x in inputs:
                    x['prompt'][0]['content'][2]['text'] = x['prompt'][0]['content'][2]['text'].format(ground_truth_emotion=None)
                
        videos_name = [x['video_name'] for x in inputs]
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)[0]
        
      #  print(prompts_text, device, '\n\n\n\n\n\n\n')
        use_audio_in_video = inputs[0].get("use_audio_in_video", False)
      #  print(len(prompts_text), prompts_text)

        images = [item for sublist in inputs  for item in sublist["images"]] if inputs[0]['images'] is not None else None
        audios = [item for sublist in inputs  for item in sublist["audios"]] if inputs[0]['audios'] is not None else None
        videos = [item for sublist in inputs  for item in sublist["videos"]] if inputs[0]['videos'] is not None else None
        
      #  print(prompts_text)
        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            audios,
            videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            use_audio_in_video=use_audio_in_video,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs["use_audio_in_video"] = True

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generate_returned_result = unwrapped_model.generate(
                **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                generation_config=self.generation_config
            )
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if self.llm_to_instruct:
            prompt_emotion = """
                You are an AI assistant tasked with inferring the most likely emotion category based on the description of a video. The description includes details about the visual content, audio information (including ASR transcripts), tone, pace, and other observations.

                Description: {think_content}

                Based on the above description, infer the most likely emotion category from the following list:
                - happy
                - surprise
                - neutral
                - angry
                - disgust
                - sad
                - fear
                - contemptuous
                - disappointed
                - helpless
                - anxious

                **Important Rules:**
                1. Select ONLY ONE emotion category that best matches the description.
                2. Do NOT generate additional text or explanations; only output the selected emotion word.
            """
  
            prompt_emotion = prompt_emotion.format(think_content=completions[0])
            messages_emotion = [
                {"role": "user", "content": prompt_emotion}
            ]
            text_emotion = self.revision_tokenizer.apply_chat_template(
                messages_emotion,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs_emotion = self.revision_tokenizer([text_emotion], return_tensors="pt").to(self.revision_model.device)

                # 2. 模型生成文本
            generated_ids_emotion = self.revision_model.generate(
                **model_inputs_emotion,
                max_new_tokens=512
            )

            # 3. 提取生成部分
            generated_ids_emotion = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs_emotion.input_ids, generated_ids_emotion)
            ]

            # 4. 解码生成的 token ID
            completions_emotion = self.revision_tokenizer.batch_decode(generated_ids_emotion, skip_special_tokens=True)[0]
          #  print("!!!!!", completions)
            completions_tmp = []
            for completion in completions:
                completion_t = f"<think>{completion}</think><answer>{completions_emotion}</answer>"
                completions_tmp.append(completion_t)
                completions = completions_tmp
          #  print("#####", completions)

        if self.with_revision_omni:
            content_match = re.search(r'\s*<think>\s*(.*?)\s*</think>\s*', completions[0], re.DOTALL)

            student_answer = content_match.group(1).strip() if content_match else completions[0].strip()
            prompt_rating = """
                You are an AI assistant tasked with evaluating the quality of a video description. The description includes visual content, audio information (including ASR transcripts), tone, pace, and other observations.

                **Core Evaluation Requirement**:  
                Accuracy is the **primary criterion** (weighted at 50%). Prioritize evaluating whether the description accurately reflects the video content. If accuracy is insufficient, even if other dimensions perform well, the overall score should be significantly reduced.

                Description: {description}

                Please rate the description based on the following five dimensions (0–1 scale):
                1. **Accuracy (50% weight)** - Does the description precisely reflect the video content? (0.0=seriously deviated, 1.0=perfect match)
                2. **Coherence (15% weight)** - Is the description logically structured and easy to follow? (0.0=incoherent, 1.0=flawless flow)
                3. **Completeness (15% weight)** - Does it cover key elements (visual, audio, emotional cues)? (0.0=missing critical elements, 1.0=exhaustive)
                4. **Clarity (15% weight)** - Are the observations specific and detailed? (0.0=vague/general, 1.0=precise/detailed)
                5. **Relevance (5% weight)** - Does the analysis focus on emotion-related features? (0.0=off-topic, 1.0=perfectly targeted)

                **Important Rules**:
                1. Output a single numerical score between 0 and 1, rounded to two decimal places.
                2. Do NOT generate additional text or explanations; only output the score.
                3. If the accuracy score is below 0.6, use the accuracy score directly as the final result, ignoring other dimensions.
            """

            inputs_copy = copy.deepcopy(inputs)
            for x in inputs_copy:
                x['prompt'][0]['content'][2]['text'] = prompt_rating.format(description=student_answer)

            videos_name = [x['video_name'] for x in inputs]
            prompts_revision = [x["prompt"] for x in inputs_copy]
            prompts_text_revision = self.vlm_module.prepare_prompt(self.processing_class, inputs_copy)[0]

            prompt_inputs_revision = self.vlm_module.prepare_model_inputs(
                self.processing_class,
                prompts_text_revision,
                images,
                audios,
                videos,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                use_audio_in_video=use_audio_in_video,
            )
            prompt_inputs_revision = super()._prepare_inputs(prompt_inputs_revision)
            prompt_inputs_revision["use_audio_in_video"] = True
            prompt_ids_revision, prompt_mask_revision = prompt_inputs_revision["input_ids"], prompt_inputs_revision["attention_mask"]
            generate_revision = self.ref_model.generate(
                **{k: v for k, v in prompt_inputs_revision.items() if k not in self.vlm_module.get_non_generate_params()}, 
                generation_config=self.generation_config
            )
            prompt_length_revision = prompt_ids_revision.size(1)

            prompt_completion_ids_revision = generate_revision
            prompt_ids_revision = prompt_completion_ids_revision[:, :prompt_length_revision]
            completion_ids_revision = prompt_completion_ids_revision[:, prompt_length_revision:]
            completions_revision_omni = self.processing_class.batch_decode(completion_ids_revision, skip_special_tokens=True)
            print("#####", completions_revision_omni)
            

        if self.with_revision:
            inputs_copy = copy.deepcopy(inputs)
            return_completions = completions
            prompt_revision = """
            You are an AI assistant tasked with generating a SINGLE question-answer pair based on the description of a video. The description includes details about the visual content, audio information (including ASR transcripts), tone, pace, and other objective observations.

            Description: {think_content}

            Based on the above description, generate ONLY ONE question that asks about a **specific detail** of the video. The question should be as **granular** as possible, focusing on a single aspect such as:
            - A specific visual observation 
            - A specific phrase or sentence from the ASR transcript 
            - A specific audio feature 

            **Important Rules:**
            1. Do NOT ask questions related to emotions, feelings, or subjective interpretations of the video content.
            2. Generate ONLY ONE question-answer pair.
            3. Ensure the question is highly specific and focuses on a single detail.

            Then, provide a concise answer to the question using the information from the description. The answer should be brief and directly related to the question.

            Your response MUST be in the following JSON format:
            {{
                "question": "<your_generated_question>",
                "answer": "<your_generated_answer>"
            }}

            Generated Question-Answer Pair:
            """

            
         #   content_match = re.search(r'<think>(.*?)</think>', completions[0])

            content_match = re.search(r'\s*<think>\s*(.*?)\s*</think>\s*', completions[0], re.DOTALL)

            student_answer = content_match.group(1).strip() if content_match else completions[0].strip()
            prompt_revision = prompt_revision.format(think_content=student_answer)
            messages_revision = [
                {"role": "user", "content": prompt_revision}
            ]
            text_revision = self.revision_tokenizer.apply_chat_template(
                messages_revision,
                tokenize=False,
                add_generation_prompt=True
            )

            response_revision = self.generate_valid_json(text_revision, solution=inputs_copy[0]['solution'])
           # print(response_revision, self.revision_model.device)
            # print(response_revision, inputs, self.revision_model.device)
            if len(response_revision.keys()) != 0:
                for x in inputs_copy:
                    x['prompt'][0]['content'][2]['text'] = response_revision['question']
                    x['solution'] = response_revision['answer']
                videos_name = [x['video_name'] for x in inputs]
                prompts_revision = [x["prompt"] for x in inputs_copy]
                prompts_text_revision = self.vlm_module.prepare_prompt(self.processing_class, inputs_copy)[0]

                prompt_inputs_revision = self.vlm_module.prepare_model_inputs(
                    self.processing_class,
                    prompts_text_revision,
                    images,
                    audios,
                    videos,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                    use_audio_in_video=use_audio_in_video,
                )
                prompt_inputs_revision = super()._prepare_inputs(prompt_inputs_revision)
                prompt_inputs_revision["use_audio_in_video"] = True
                prompt_ids_revision, prompt_mask_revision = prompt_inputs_revision["input_ids"], prompt_inputs_revision["attention_mask"]
                generate_revision = self.ref_model.generate(
                    **{k: v for k, v in prompt_inputs_revision.items() if k not in self.vlm_module.get_non_generate_params()}, 
                    generation_config=self.generation_config
                )
                prompt_length_revision = prompt_ids_revision.size(1)

                prompt_completion_ids_revision = generate_revision
                prompt_ids_revision = prompt_completion_ids_revision[:, :prompt_length_revision]
                completion_ids_revision = prompt_completion_ids_revision[:, prompt_length_revision:]
                completions_revision = self.processing_class.batch_decode(completion_ids_revision, skip_special_tokens=True)
            
        if self.logical_withllm:
            prompt_emotion = """
                You are an AI assistant tasked with inferring the most likely emotion category based on the description of a video. The description includes details about the visual content, audio information (including ASR transcripts), tone, pace, and other observations.

                Description: {think_content}

                Based on the above description, infer the most likely emotion category from the following list:
                - happy
                - surprise
                - neutral
                - angry
                - disgust
                - sad
                - fear
                - contemptuous
                - disappointed
                - helpless
                - anxious

                **Important Rules:**
                1. Select ONLY ONE emotion category that best matches the description.
                2. Do NOT generate additional text or explanations; only output the selected emotion word.
            """

          #  content_match_logical = re.search(r'<think>(.*?)</think>', completions[0])

            content_match_logical = re.search(r'\s*<think>\s*(.*?)\s*</think>\s*', completions[0], re.DOTALL)

            student_answer_logical = content_match_logical.group(1).strip() if content_match_logical else completions[0].strip()
          #  predict_match = re.search(r'<answer>(.*?)</answer>', completions[0])
            predict_match = re.search(r'\s*<answer>\s*(.*?)\s*</answer>\s*', completions[0], re.DOTALL)
            predict_emotion = predict_match.group(1).strip() if predict_match else ""
            predict_emotion = [predict_emotion]
  
            prompt_emotion = prompt_emotion.format(think_content=student_answer_logical)
            messages_emotion = [
                {"role": "user", "content": prompt_emotion}
            ]
            text_emotion = self.revision_tokenizer.apply_chat_template(
                messages_emotion,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs_emotion = self.revision_tokenizer([text_emotion], return_tensors="pt").to(self.revision_model.device)

                # 2. 模型生成文本
            generated_ids_emotion = self.revision_model.generate(
                **model_inputs_emotion,
                max_new_tokens=512
            )

            # 3. 提取生成部分
            generated_ids_emotion = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs_emotion.input_ids, generated_ids_emotion)
            ]

            # 4. 解码生成的 token ID
            completions_emotion = self.revision_tokenizer.batch_decode(generated_ids_emotion, skip_special_tokens=True)



        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            if self.with_revision:
                completions_revision = [[{"role": "assistant", "content": completion}] for completion in completions_revision]
            if self.logical_withllm:
                completions_emotion = [[{"role": "assistant", "content": completion}] for completion in completions_emotion]
        
        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt
        print(videos_name)
        if not self.with_revision_omni:
            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        else:
            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs) + 1, device=device)

        if not (self.with_revision or self.logical_withllm):
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, PreTrainedModel):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                    for key in reward_kwargs:
                        for example in inputs:
                            # No need to duplicate prompts as we're not generating multiple completions per prompt
                            # reward_kwargs[key].extend([example[key]] * self.num_generations)
                            reward_kwargs[key].extend([example[key]])
                    output_reward_func = reward_func(prompts=prompts, completions=completions, tokenizer=self.processing_class, embedding_layer=self.embedding_layer,**reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        else:
            for i, (reward_func) in enumerate(self.reward_funcs):
                # print(reward_func.__name__, len(self.reward_funcs), i, sellf.func_ids)
                # ddd
                if self.func_ids[i] == 'origin':
                    reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                    for key in reward_kwargs:
                        for example in inputs:
                            # No need to duplicate prompts as we're not generating multiple completions per prompt
                            # reward_kwargs[key].extend([example[key]] * self.num_generations)
                            reward_kwargs[key].extend([example[key]])
                    # print(reward_kwargs, inputs, reward_func)
                    # ddd
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                if self.func_ids[i] == 'revision':
                    if len(response_revision.keys()) != 0:
                        reward_kwargs = {key: [] for key in inputs_copy[0].keys() if key not in ["prompt", "completion"]}
                        for key in reward_kwargs:
                            for example in inputs_copy:
                                # No need to duplicate prompts as we're not generating multiple completions per prompt
                                # reward_kwargs[key].extend([example[key]] * self.num_generations)
                                reward_kwargs[key].extend([example[key]])
                        output_reward_func = reward_func(prompts=prompts_revision, completions=completions_revision, **reward_kwargs)
                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                    else:
                        output_reward_func = [0]
                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                if self.func_ids[i] == 'logical':
                    output_reward_func = reward_func(completions=completions_emotion, solution=predict_emotion)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if self.with_revision_omni:
            try:
                completions_revision_omni = float(completions_revision_omni[0])
            except (ValueError, TypeError, IndexError):
                completions_revision_omni = 0.5  # 默认值
            score_omni = [completions_revision_omni]
            rewards_per_func[:, -1] = torch.tensor(score_omni, dtype=torch.float32, device=device)
            print(rewards_per_func)
        
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
      #  print("22222", rewards_per_func, "\n\n\n\n\n")
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

      #  print("33333", rewards, "\n\n\n\n\n")
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
       # print(rewards, mean_grouped_rewards)
      #  ddd
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        if self.with_revision:
            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "old_per_token_logps": old_per_token_logps,
                "ref_per_token_logps": ref_per_token_logps,
                "advantages": advantages,
                "videos_name": videos_name,
                "multimodal_inputs": multimodal_inputs,
                "completions": return_completions
            }
        else:
            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "old_per_token_logps": old_per_token_logps,
                "ref_per_token_logps": ref_per_token_logps,
                "advantages": advantages,
                "videos_name": videos_name,
                "multimodal_inputs": multimodal_inputs
            }

    def generate_valid_json(self, text_revision, solution):
        """
        生成符合 JSON 格式的文本，如果生成的内容不符合格式，则重新生成，直到成功为止。
        """
        max_attempts = 100  # 设置最大尝试次数，避免无限循环
        attempt = 0
  
        while attempt < max_attempts:
            try:
                # 1. 准备输入数据
                model_inputs_revision = self.revision_tokenizer([text_revision], return_tensors="pt").to(self.revision_model.device)

                # 2. 模型生成文本
                generated_ids_revision = self.revision_model.generate(
                    **model_inputs_revision,
                    max_new_tokens=512
                )

                # 3. 提取生成部分
                generated_ids_revision = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs_revision.input_ids, generated_ids_revision)
                ]

                # 4. 解码生成的 token ID
                response_revision = self.revision_tokenizer.batch_decode(generated_ids_revision, skip_special_tokens=True)[0]
             #   print("!!!!", response_revision)
                # 5. 尝试解析为 JSON 字典
                response_dict = json.loads(response_revision)

                # 如果解析成功，返回结果
                return response_dict

            except json.JSONDecodeError:
                # 如果解析失败，记录日志并重试
                print(f"Attempt {attempt + 1} failed: Generated text is not valid JSON. Retrying...")
                return {"question": "What's the most obvious emotion in the video?",
                 "answer": solution}

        # 如果达到最大尝试次数仍未成功，抛出异常或返回默认值
        raise ValueError("Failed to generate valid JSON after maximum attempts.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        device = self.accelerator.device
       # print(device)
        
        # if device == torch.device('cuda:0'):
        #     print("!!!!!!!", inputs_copy[0].keys())
        # Check if we need to generate new completions or use buffered ones
      #  print(self.state.global_step, self.num_iterations)
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1
        # if device == torch.device('cuda:0'):
        #     print("######", inputs_copy[0].keys(), inputs.keys())


            

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]
    #    print(per_token_logps)
        prob_sentence = torch.mean(per_token_logps, dim=1)
        
  

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        if self.only_deterministic:
            if device != torch.device('cuda:0'):
                advantages = 0.0 * advantages


     #   advantages = advantages * prob_sentence
     #   print(prob_sentence, advantages)
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        # print("111",advantages)
        # print("222",per_token_logps, old_per_token_logps)
        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        # print(coef_1, coef_2)
        # print(per_token_loss)
        

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
           # per_token_loss = per_token_loss + self.beta * per_token_kl
            if self.use_drgrpo:
                per_token_loss = per_token_loss
            else:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        # print(per_token_loss, completion_mask)
        # ddd
        if self.use_drgrpo:
            loss = 0.1 * ((per_token_loss * completion_mask).sum(dim=1)).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
     #   print(loss)
     #   ddd
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
