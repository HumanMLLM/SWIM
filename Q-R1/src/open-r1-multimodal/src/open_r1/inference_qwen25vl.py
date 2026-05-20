from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from data_process.vision_process import process_vision_info
import torch
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

class AttentionVisualizer:
    def __init__(self, model, grid_thw=None):
        """
        model: Qwen2_5_VisionTransformerPretrainedModel 实例
        grid_thw: Tensor [num_videos, 3] (T,H,W)，用于恢复 token 位置
        """
        self.model = model
        self.grid_thw = grid_thw
        self.attn_maps = {}  # {layer_idx: attn_probs}

        self.hooks = []
        # 注册 hook 到全局 attention 层
        for layer_idx, blk in enumerate(model.blocks):
            if layer_idx in model.fullatt_block_indexes:
                handle = blk.attn.register_forward_hook(self._save_attn_hook(layer_idx))
                self.hooks.append(handle)

    def _save_attn_hook(self, layer_idx):
        def fn(module, input, output):
            # 这里假设 attention 模块有 attn_probs/attn_weights 变量存注意力
            if hasattr(module, "attn_probs"):
                attn = module.attn_probs.detach().cpu()
            elif hasattr(module, "attn_weights"):
                attn = module.attn_weights.detach().cpu()
            else:
                raise ValueError("Attention module does not expose attn_probs or attn_weights.")
            self.attn_maps[layer_idx] = attn
        return fn

    def clear(self):
        """清空保存的 attention map"""
        self.attn_maps.clear()

    def remove_hooks(self):
        """移除 hook，释放资源"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def visualize(self, layer_idx, head=0, query_token=0, merge_size=None, save_path=None):
        """
        layer_idx: 目标层编号（全局 attention）
        head: 选择的注意力头
        query_token: Query token 的索引
        merge_size: spatial merge size（e.g., 2），用于恢复到 (H,W)
        save_path: 是否保存热力图
        """
        if layer_idx not in self.attn_maps:
            raise ValueError(f"No attention map for layer {layer_idx}. Did you run a forward pass?")
        
        attn = self.attn_maps[layer_idx]  # shape (batch, num_heads, Q_len, K_len)
        attn_map = attn[0, head, query_token]  # 取 batch=0 样本

        if self.grid_thw is not None:
            # 恢复到空间形状
            T, H, W = self.grid_thw[0].tolist()
            if merge_size is not None:
                H //= merge_size
                W //= merge_size
            # 只取第一帧可视化
            attn_frame0 = attn_map[:H*W].reshape(H, W)
            heatmap = attn_frame0.numpy()
        else:
            # 直接 reshape 尝试
            side = int(np.sqrt(attn_map.shape[0]))
            heatmap = attn_map[:side*side].reshape(side, side).numpy()

        plt.imshow(heatmap, cmap='hot')
        plt.colorbar()
        plt.title(f"Layer {layer_idx} Head {head} Token {query_token}")
        if save_path:
            plt.savefig(save_path)
        plt.show()

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/<MODEL_PATH>", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "<MODEL_PATH>",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processor
processor = AutoProcessor.from_pretrained("<MODEL_PATH>")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/<MODEL_PATH>", min_pixels=min_pixels, max_pixels=max_pixels)


# json_file = "./data/DAMO-NLP-SG/VideoRefer-700K/format-videorefer-detailed-caption-125k.json"
# data_dict = json.load(open(json_file))
# data_sample = data_dict[0]

# video_path = './data/DAMO-NLP-SG/VideoRefer-700K/video_refer_detailed_caption_mask_all_frame/' + data_sample['index']
# video_frames = os.listdir(video_path)
# video_frames = [os.path.join(video_path, vf) for vf in video_frames]

image = './data/20260126-180405.jpg'

prompt = "Can you discuss in detail the important elements of the man walking on the street in the video "
print(image, prompt)
messages = [
    {
        "role": "user",
        "content": [
            # {"type": "video", "video":video_frames},
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(text)
# exit()
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128, output_attentions_vision=False, output_attentions=True)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)