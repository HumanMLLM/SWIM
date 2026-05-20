#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理 HC-STVG 视频，统一 prompt：describe the person in the video
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

PROMPT = "Can you discuss in detail the important elements of the person in the video?"

# ---------------- 数据读取 ----------------
json_file = Path("./data/HC-STVG/val_no_bbox_with_path.json")
data = json.loads(json_file.read_text(encoding="utf-8"))

# ---------------- 批量推理 ----------------
for filename, item in tqdm(data.items(), desc="inferencing"):
    video_path = item.get("video_path")
    if not video_path or not Path(video_path).exists():
        item["model_pred"] = "[Video not found]"
        continue

    # 1. 抽帧（这里每 8 帧取 1 帧，可改）
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in range(0, total, fps // 2):   # 0.5 秒一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = cap.read()
        if ret:
            # 转 RGB，保存为临时文件列表（模型要求路径列表）
            tmp = f"/tmp/frm_{idx}.jpg"
            cv2.imwrite(tmp, frm)
            frames.append(tmp)
    cap.release()

    if not frames:
        item["model_pred"] = "[Empty video]"
        continue
    print(f"Extracted {len(frames)} frames from {video_path}")
    # 2. 构造 message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    # 3. 推理
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
    print(f"{filename} → {item['model_pred']}")

    # 4. 清理临时帧
    for f in frames:
        os.remove(f)

# ---------------- 写回 ----------------
out_file = out_file = Path("./data/HC-STVG/val_no_bbox_with_path_pred.json")
out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ 已保存 → {out_file}")