#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理 Ref-L4 图片，统一 prompt：describe the person in the image
"""
import json
import os
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from data_process.vision_process import process_vision_info

# ---------------- 模型初始化 ----------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "<MODEL_PATH>",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("<MODEL_PATH>")

PROMPT = "Can you discuss in detail the important elements of the highlighted object highlighted by the red box in the image?"
# PROMPT = "Give the caption of the image focusing on the highlighted object. Do not mention the bounding box itself. An example answer: Within the central picture frame of the three, an antique camera is present."
# ---------------- 数据读取 ----------------
json_file = Path("./data/Ref-L4/ref-l4-val-no-bbox-formatted_with_abs_path_checked_new.json")
data = json.loads(json_file.read_text(encoding="utf-8"))

# ---------------- 批量推理 ----------------
for item in tqdm(data, desc="inferencing"):
    if not item.get("is_rewrite", False):
        item["model_pred"] = "[Image not found]"
        continue

    img_path = item["file_name"]
    if not Path(img_path).is_file():
        item["model_pred"] = "[Image missing]"
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    pred: str = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    item["model_pred"] = pred.strip()
    print(f"{img_path} → {item['model_pred']}")

# ---------------- 写回 ----------------
out_file = json_file.with_name(json_file.stem + "_pred_2.json")
out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ 已保存 → {out_file.resolve()}")