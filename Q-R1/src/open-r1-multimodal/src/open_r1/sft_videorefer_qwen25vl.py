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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
import yaml
import json
import math
import random
from PIL import Image
import pycocotools.mask as maskUtils
import re
from collections import defaultdict

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from dataclasses import field
from data_process.vision_process import process_vision_info


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

logger = logging.getLogger(__name__)
from dataclasses import dataclass

import builtins as __builtin__
import torch.distributed as dist

def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 3:
        __builtin__.print(*args, **kwargs)

def annToMask(rle):
    m = maskUtils.decode(rle)
    return m

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})


processor = None

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: ScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                #     data_root: xxxx/xx

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None
                    print("sampling_strategy", sampling_strategy)

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

    

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        print("Sampling first: ", sampling_number)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    if data.get("data_root", None):
                        for each in cur_data_dict:
                            if "path" in each:
                                each["path"] = os.path.join(data["data_root"], each["path"])
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        self.mel_size = 128
        self.frames_upbound = 16

    def __len__(self):
        return len(self.list_data_dict)


    def _maybe_apply_format_convert_videorefer(self, source):
        full_video_path = source['video']
        if 'videorefer_short_caption' in source['index']:
            video_path = './data/DAMO-NLP-SG/VideoRefer-700K/video_refer_short_caption_mask_all_frame/' + source['index']
            # full_video_path = source['video']
            prompt = 'I have outlined an object with a red contour in the first frame in the video. '
        elif 'videorefer_detailed_caption' in source['index']:
            video_path = './data/DAMO-NLP-SG/VideoRefer-700K/video_refer_detailed_caption_mask_all_frame/' + source['index']
            # full_video_path = source['video']
            prompt = 'I have outlined an object with a red contour in the video. '

        use_full_video = True
        if not use_full_video:
            video_frames = os.listdir(video_path)
            video_frames = [os.path.join(video_path, vf) for vf in video_frames]
            prompt += source['conversations'][0]['value'].replace('<region>','')
        else:
            video_full = full_video_path
            prompt = source['conversations'][0]['value']
            all_masks = source['annotation']
            for idx, cur_instance in enumerate(all_masks):
                for instance_mask in cur_instance:
                    real_mask = annToMask(cur_instance[instance_mask]['segmentation'])
                    all_masks[idx][instance_mask]['mask'] = real_mask

        if use_full_video:
            if all_masks == [] or 'qa' in source['index']:
                conversation = [
                {
                    "role": "system",
                    "content": "You are Qwen, a helpful assistant with global scene understanding, capable of grounding user-specified text to the corresponding object instance in the image and providing fine-grained analysis based on visual evidence.",
                },
                { 
                        "role": "user",
                        "content":[
                            {"type": "video", "video":video_full},
                            {"type": "text", "text": prompt.replace('<ins>', '').replace('</ins>', '')},
                        ]
                },
                {'role': 'assistant', 'content': source['conversations'][1]['value'].replace('<ins>', '').replace('</ins>', '')}
                ]
            else:
                conversation = [
                {
                    "role": "system",
                    "content": "You are Qwen, a helpful assistant with global scene understanding, capable of grounding user-specified text to the corresponding object instance in the image and providing fine-grained analysis based on visual evidence.",
                },
                { 
                        "role": "user",
                        "content":[
                            {"type": "video", "video":video_full},
                            {"type": "text", "text": prompt},
                            {"type": "object_masks", "object_masks": all_masks},
                        ]
                },
                {'role': 'assistant', 'content': source['conversations'][1]['value']}
                ]
            # print(conversation)
            # exit()
        else:
            conversation = [
            {
                "role": "system",
                "content": "You are Qwen, a helpful assistant with global scene understanding, capable of grounding user-specified text to the corresponding object instance in the image and providing fine-grained analysis based on visual evidence.",
            },
            { 
                    "role": "user",
                    "content":[
                        {"type": "video", "video":video_frames},
                        {"type": "text", "text": prompt},
                    ]
            },
            {'role': 'assistant', 'content': source['conversations'][1]['value']}
            ]            
        return conversation


    def _maybe_apply_format_convert_llava_single_turn(self, source):
        # 判断模态
        if 'video' in source:
            data_type = 'video'
            raw_path = source['video']
        elif 'image' in source:
            data_type = 'image'
            raw_path = source['image']
        else:
            raise ValueError("source must contain 'video' or 'image' field")

        convs = source.get('conversations', [])
        if len(convs) < 2:
            raise ValueError(f"Not enough conversation turns in source: {source}")

        prompt = source['conversations'][0]['value']

        if data_type == 'video':

            conversation = [
            {
                "role": "system",
                "content": "You are Qwen, a helpful assistant with global scene understanding, capable of grounding user-specified text to the corresponding object instance in the image and providing fine-grained analysis based on visual evidence.",
            },
                { 
                    "role": "user",
                    "content":[
                            {"type": "video", "video":raw_path},
                            {"type": "text", "text": prompt},
                        ]
                },
                {'role': 'assistant', 'content': source['conversations'][1]['value']}
                ]    
        elif data_type == 'image':
            conversation = [
            {
                "role": "system",
                "content": "You are Qwen, a helpful assistant with global scene understanding, capable of grounding user-specified text to the corresponding object instance in the image and providing fine-grained analysis based on visual evidence.",
            },
                { 
                    "role": "user",
                    "content":[
                            {"type": "image", "image":raw_path},
                            {"type": "text", "text": prompt},
                        ]
                },
                {'role': 'assistant', 'content': source['conversations'][1]['value']}
                ]        
        return conversation

    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 3
        import traceback

        try:
            return self._get_item(i)
        except Exception as e:
            print(i)
            traceback.print_exc()


        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                traceback.print_exc()
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        
    def _get_item(self, i):
        source = self.list_data_dict[i]
        if 'index' in source and 'videorefer' in source['index']:
            messages = self._maybe_apply_format_convert_videorefer(source)
        elif 'data_source' in source and 'LLaVA-Video-178K' in source['video']:
            messages = self._maybe_apply_format_convert_llava_single_turn(source)
        else:
            print(source)
            exit()

        return {
            'messages': messages,
        #  
        }

def extract_ins_with_global_occurrence(text):
    """
    提取 <ins> 中的内容，并记录它在整个文本中是第几次出现（计数从1开始）

    Args:
        text (str): 原始文本

    Returns:
        ins_contents (list[str]): <ins> 中的内容列表
        occurrences (list[int]): 对应内容在整个文本中出现的次数（第几次出现）
        clean_text (str): 去掉 <ins> 标签后的文本
    """
    # 去掉 <ins> 标签
    clean_text = re.sub(r"</?ins>", "", text)
    
    # 提取 <ins> 中的内容
    matches = re.finditer(r"<ins>(.*?)</ins>", text)
    ins_contents = []
    for m in matches:
        content = m.group(1)
        start = m.start()
        # 如果 <ins> 前有空格，则在内容前加空格
        if start > 0 and text[start - 1] == " ":
            content = " " + content
        ins_contents.append(content)
    
    # 遍历文本，统计每个词出现次数
    words = re.findall(r"\w+", text)
    counter = defaultdict(int)
    word_occurrences = defaultdict(list)  # 记录每个词的出现顺序
    for i, word in enumerate(words, 1):  # i 从1开始计数
        counter[word] += 1
        word_occurrences[word].append(counter[word])

    # 对每个 <ins> 内容，取它在文本中的第几次出现（顺序）
    occurrences = []
    current_count = defaultdict(int)
    for word in ins_contents:
        current_count[word] += 1
        occurrences.append(current_count[word])

    return ins_contents, occurrences, clean_text

def collate_fn(examples):
    # print_rank0(examples[0]["messages"])
    # exit(0)
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        for example in examples
    ]
    # print_rank0(texts)
    ins_contents = []
    ins_occurrences = []
    clean_texts = []
    for text in texts:
        ins_content, ins_occurrence, clean_text = extract_ins_with_global_occurrence(text)
        ins_contents.append(ins_content)
        ins_occurrences.append(ins_occurrence)
        clean_texts.append(clean_text)
    # print_rank0(ins_contents, ins_occurrences, clean_texts)
    # exit()
    texts = clean_texts
    # print_rank0(texts)  
    # exit()

    masks = None
    # print_rank0(examples[0]["messages"][1]["content"])
    # 检查数据结构是否符合预期，并且存在object_masks字段且不为None
    try:
        first_msg_content = examples[0]["messages"][1]["content"]

        if len(first_msg_content) > 2 and \
        isinstance(first_msg_content[2], dict) and \
        "object_masks" in first_msg_content[2] and \
        first_msg_content[2]["object_masks"] is not None:
        
            masks = [
                example["messages"][1]["content"][2]["object_masks"]
                for example in examples
                if len(example["messages"][1]["content"]) > 2 and
                isinstance(example["messages"][1]["content"][2], dict) and
                "object_masks" in example["messages"][1]["content"][2]
            ]
    except (IndexError, KeyError, TypeError) as e:
        # 数据结构不符合预期，保持 masks = None
        print(f"[警告] JSON 中缺少 object_masks字段或结构不符: {e}")
        pass

    # print_rank0(masks)
    # exit()

    images = []
    videos = []
    audios = []
    mask_indexs = []
    # print_rank0("examples ", examples[0]["messages"], examples[1]["messages"])
    # exit()
    for example in examples:
        # print_rank0(example["messages"])
        # exit()
        if masks is not None:
            image, video, mask_index = process_vision_info(example["messages"])
        else:
            image, video = process_vision_info(example["messages"])
        if image is not None: 
            images.extend(image)
        if video is not None: 
            videos.extend(video)
        if masks is not None and mask_index is not None:
            # print_rank0("mask_index per sample ", mask_index)
            # import pdb
            # pdb.set_trace()
            mask_indexs.append(mask_index)   
    if len(videos)==0:
        videos = None
    if len(images)==0:
        images = None
    # print_rank0("video[0].shape ", video[0].shape)
    # exit()
    # 此处video shape 为 [B, frames, 3, h, w]
    print_rank0("texts ", texts)
    batch = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            # fps=1.0
        )

    sep = processor.tokenizer("<|im_start|>assistant\n")['input_ids']
    labels = batch["input_ids"].clone()

    # 找到 sep 在 input_ids 中的起始位置
    batch_size, seq_length = batch["input_ids"].shape
    sep_positions = []
    for i in range(batch_size):
        # 遍历每个样本，找到 sep 的起始位置
        for j in range(seq_length - len(sep) + 1):
            if torch.equal(batch["input_ids"][i, j:j+len(sep)], torch.tensor(sep)):
                sep_positions.append(j)
                break
        else:
            # 如果未找到 sep，则默认设置为序列末尾
            sep_positions.append(seq_length)

    # 将 sep 之前的所有 labels 设置为 -100
    for i, sep_pos in enumerate(sep_positions):
        labels[i, :sep_pos] = -100

    # 对 sep 之后的部分进行处理
    labels[labels == processor.tokenizer.pad_token_id] = -100  # 将 pad_token 对应的 labels 设置为 -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100  # 将 image_token 对应的 labels 设置为 -100
    num_l = (labels!=-100)
    # 更新 batch 中的 labels    
    batch["labels"] = labels


    sep_attn = processor.tokenizer("<|vision_end|>")['input_ids']
    attn_labels = batch["input_ids"].clone()
    sep_attn_positions = []
    for i in range(batch_size):
        for j in range(seq_length - len(sep_attn) + 1):
            if torch.equal(batch["input_ids"][i, j:j+len(sep_attn)], torch.tensor(sep_attn)):
                sep_attn_positions.append(j)
                break
        else:
            sep_attn_positions.append(seq_length)
    
    for i, sep_pos in enumerate(sep_attn_positions):
        attn_labels[i, :sep_pos+1] = -100

    attn_labels[attn_labels == processor.tokenizer.pad_token_id] = -101
    attn_labels[attn_labels == image_token_id] = -101
    batch["attn_labels"] = attn_labels


    sep_ins_contents = []
    ins_contents_labels = torch.full_like(batch["input_ids"], -101)
    for ins_content, ins_occurrence in zip(ins_contents, ins_occurrences):
        sep_ins_content = [processor.tokenizer(cur_ins_content)['input_ids'] for cur_ins_content in ins_content]
        sep_ins_contents.append(sep_ins_content)

    for i in range(batch_size):
        cnt_ins_list = [0] * len(sep_ins_contents[i])
        for j in range(seq_length):
            for cur_ins_ids in sep_ins_contents[i]:
                if torch.equal(batch["input_ids"][i, j:j+len(cur_ins_ids)], torch.tensor(cur_ins_ids)):
                    cur_ins_idx = sep_ins_contents[i].index(cur_ins_ids)
                    cnt_ins_list[cur_ins_idx] += 1
                    if cnt_ins_list[cur_ins_idx] == ins_occurrences[i][cur_ins_idx]:
                        ins_contents_labels[i,j:j+len(cur_ins_ids)] = cur_ins_idx
                

        
    
    if masks is not None:
        # 视频所有mask
        batch["ins_masks"] = masks
        # 哪些采样帧有mask
        batch['mask_index'] = mask_indexs
        # 在何处计算attention loss
        batch["ins_contents_labels"] = ins_contents_labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower() or "omni" in model_args.model_name_or_path.lower():

        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    vision_modules_keywords = ['visual','audio_tower']
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "qwen" in model_args.model_name_or_path.lower() and "omni" in model_args.model_name_or_path.lower():
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model.thinker.config.vocab_size = 152064
        model = model.thinker
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    print("Freezing vision modules...")
    for n, p in model.named_parameters():
        if any(keyword in n for keyword in vision_modules_keywords):
            # print_rank0(n, p)
            p.requires_grad = False
    # exit()
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        # "finetuned_from": model_args.model_name_or_path,
        # "dataset": list(script_args.dataset_name),
        "dataset_name": list(script_args.dataset_name),
        # "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)




if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)