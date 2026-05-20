import os
import json
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from qwen_vl_utils import process_vision_info

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLProcessor

COT_BRIEF_PROMPT = "{}\nAnswer the question using a single word or phrase."
torch.set_printoptions(profile="full")
def parse_args():
    parser = argparse.ArgumentParser(description="Save attention weights from Qwen2.5 VL model.")
    parser.add_argument("--vision_info_path", default="./data/DAMO-NLP-SG/VideoRefer-700K/format-videorefer-detailed-caption-125k-part.json", type=str, help="Path to the vision info file.")
    parser.add_argument("--output_dir",default="./vis/attn_vis", type=str, help="Directory to save the attention weights.")
    parser.add_argument("--model_path", type=str, default="<MODEL_PATH>")
    # parser.add_argument("--output_dir",default="./Q-R1/src/open-r1-multimodal/src/open_r1/attn_vis/<MODEL_PATH>/attn_refine", type=str, help="Directory to save the attention weights.")
    # parser.add_argument("--model_path", type=str, default="<MODEL_PATH>")
    parser.add_argument("--layer", type=int, default=3, help="Layer index to save attention weights from.")
    parser.add_argument("--head", type=int, default=-1, help="Head index to save attention weights from.")
    parser.add_argument("--color_map", type=str, default="jet", help="Color map for attention visualization.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha for blending attention maps with images.")
    parser.add_argument("--brief", action="store_true", help="If set, use brief prompt")
    return parser.parse_args()

def prepare_labels_from_input_ids_second_last(input_ids, im_start_id):
    B, L = input_ids.shape
    labels = input_ids.clone()
    
    # 找到所有 <|im_start|> 的位置
    mask = input_ids == im_start_id
    
    # 翻转序列
    flipped_mask = mask.flip(dims=(1,))
    
    # 转成 int，便于累加
    flipped_int = flipped_mask.int()
    
    # 找到倒数第二个 <|im_start|>
    # 方法：cumsum 累加 True 的个数，找到第二个为 1 的位置
    cumsum_flipped = torch.cumsum(flipped_int, dim=1)
    second_idx_in_flipped = torch.argmax((cumsum_flipped == 2).int(), dim=1)
    
    # 如果序列中只有一个 <|im_start|>，则 fallback 到最后一个
    only_one_mask = (flipped_int.sum(dim=1) == 1)
    second_idx_in_flipped = torch.where(only_one_mask, torch.argmax(flipped_int, dim=1), second_idx_in_flipped)
    
    last_pos = (L - 1) - second_idx_in_flipped
    
    mask_until_idx = last_pos + 3
    mask_until_idx = torch.clamp(mask_until_idx, max=L)
    
    arange_l = torch.arange(L, device=input_ids.device).expand(B, -1)
    modification_mask = arange_l < mask_until_idx.unsqueeze(1)
    
    labels[modification_mask] = -100  # ignore index of CrossEntropyLoss
    return labels
def prepare_labels_from_input_ids(input_ids, im_start_id):
    B, L = input_ids.shape
    labels = input_ids.clone()
    mask = input_ids == im_start_id
    flipped_mask = mask.flip(dims=(1,))  # Reverse the mask to find the last <|im_start|> token
    first_idx_in_flipped = torch.argmax(flipped_mask.int(), dim=1)
    last_pos = (L - 1) - first_idx_in_flipped
    if im_start_id != 151644:
        mask_until_idx = last_pos + 1
    else:
        mask_until_idx = last_pos + 3
    mask_until_idx = torch.clamp(mask_until_idx, max=L)
    
    arange_l = torch.arange(L, device=input_ids.device).expand(B, -1)
    modification_mask = arange_l < mask_until_idx.unsqueeze(1)
    
    labels[modification_mask] = -100   # ignore index of CrossEntropyLoss
    return labels

def reduce_tensor(src: torch.Tensor, select_idx, dim, keepdim=False):
    if select_idx == -1:
        return src.mean(dim=dim, keepdim=keepdim)
    else:
        rtn = src.select(dim, select_idx)
        if keepdim:
            rtn = rtn.unsqueeze(dim)
        return rtn

# def save_attentions_on_image(image_path, attentions, label_ids, grid_hw, output_dir, color_map="jet", alpha=0.5):
    
#     image = Image.open(image_path).convert("RGB")
#     width, height = image.size
#     grid_h, grid_w = grid_hw
#     target_w, target_h = grid_w * 28, grid_h * 28
#     image = image.resize((target_w, target_h), Image.LANCZOS)

#     for idx, (one_attn, one_label_id) in enumerate(zip(attentions, label_ids)):
#         print(idx, one_label_id)
#         save_path = os.path.join(output_dir, f"{idx}_{one_label_id.item()}.png")
#         attention_img = one_attn.reshape(grid_h, grid_w)
#         # minmax norm
#         attention_img = (attention_img - attention_img.min()) / (attention_img.max() - attention_img.min())
#         # interpolate to image size (nearest neighbor)
#         attention_img = attention_img.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_h, grid_w)
#         attention_img = torch.nn.functional.interpolate(attention_img, size=(target_h, target_w), mode='nearest')
#         attention_img = attention_img.squeeze(0).squeeze(0)  # (height, width)
#         attention_img = (attention_img * 255).byte().cpu().numpy()

#         heatmap_rgba = cm.get_cmap(color_map)(attention_img)
#         heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
#         heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')
#         blended_img = Image.blend(image, heatmap_img, alpha=alpha)
#         blended_img.save(save_path)
#         print(f"Saved attention map to {save_path}")

def save_attentions_on_image(image_path, attentions, label_ids, grid_hw, output_dir, color_map="turbo", alpha=0.3):
    import cv2

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    grid_h, grid_w = grid_hw
    target_w, target_h = grid_w * 28, grid_h * 28
    image = image.resize((target_w, target_h), Image.LANCZOS)

    # 调暗原图背景
    # image = image.point(lambda p: p * 0.7)

    for idx, (one_attn, one_label_id) in enumerate(zip(attentions, label_ids)):
        save_path = os.path.join(output_dir, f"{idx}_{one_label_id.item()}.png")

        attention_img = one_attn.reshape(grid_h, grid_w)
        # min-max normalize
        attention_img = (attention_img - attention_img.min()) / (attention_img.max() - attention_img.min() + 1e-8)
        
        # bilinear 上采样更平滑
        attention_img = attention_img.unsqueeze(0).unsqueeze(0)
        attention_img = torch.nn.functional.interpolate(attention_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        attention_img = attention_img.squeeze().float().cpu().numpy()

        # 高斯平滑
        attention_img = cv2.GaussianBlur(attention_img, (0, 0), sigmaX=4)
        
        # 上色
        heatmap_rgba = cm.get_cmap(color_map)(attention_img)  # turbo/viridis 等
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')

        # 混合
        blended_img = Image.blend(image, heatmap_img, alpha=alpha)
        blended_img.save(save_path)
        print(f"Saved attention map to {save_path}")

def generate_answer(question, image_path, model, processor):
    messages = [[
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}
    ]]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, max_new_tokens=512)
        generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]  # remove input_ids part
    answer = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return answer


def main():
    args = parse_args()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path,
                                                                   torch_dtype=torch.bfloat16,
                                                                   attn_implementation="flash_attention_2",
                                                                   device_map="auto")
    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained("<MODEL_PATH>")
    im_start_id = processor.tokenizer.encode("<|im_start|>")[0]
    vision_end_id = processor.tokenizer.encode("<|vision_end|>")[0]
    # print(vision_end_id)

    print(f"Using <|im_start|> token ID: {im_start_id}")
    print(f"Using <|vision_end|> token ID: {vision_end_id}")
    # exit()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # with open(args.vision_info_path, 'r') as f:
    #     vision_info = json.load(f)

    # vision_info = vision_info[0]
    
    # for info_idx, one_info in enumerate(vision_info):
    # case 1
    # question = 'I have outlined an object with a red contour in the image. Please describe the important aspects of the marked object in the image.'
    # image_path = './data/DAMO-NLP-SG/VideoRefer-700K/video_refer_detailed_caption_mask_all_frame/videorefer_detailed_caption_0/00105.jpg'
    # answer = 'The man in the brown jacket and red and white plaid hat bends down to pick up a cat before lying down in the snow, his legs raised in the air. He has a beard and is situated in a snowy outdoor setting.'
    # answer = None

    # case 2
    # question = 'Provide a brief description of the given image.'
    # image_path = './data/datasets/Video-LLaVA/videollava_pt/llava_image/00473/004733955.jpg'
    # answer = 'a woman with a nurse cap on posing for a photo'
    # # answer = None
    
    # case 3
    # question = 'Please describe the man in the brown jacket and red and white plaid hat in the video'
    # image_path = './data/DAMO-NLP-SG/VideoRefer-700K/video_refer_detailed_caption_mask_all_frame/videorefer_detailed_caption_0/00105.jpg'
    # answer = 'The man in the brown jacket and red and white plaid hat bends down to pick up a cat before lying down in the snow, his legs raised in the air. He has a beard and is situated in a snowy outdoor setting.'
    # # answer = None

    # case 4
    question = 'Can you discuss in detail the important elements of the dog in the video?\n'
    # image_path = './Q-R1/src/open-r1-multimodal/src/open_r1/videorefer_d_vis/7/00017.jpg'
    image_path = './data/dog.jpg'
    answer = 'The young man with short, light brown hair and a beard is walking down the street.'
    
    # case 5
    # question = 'Please describe the far left side cub in the entire video in detail.'
    # image_path = './data/VideoRefer/benchmark/eval/VideoRefer-Bench-D/masked-all-frame-no-countor/0/00006.jpg'
    # answer = 'The cub is a smaller, light colored lion. It is lying down and resting its head against the other lion. The cub looks calm and relaxed. It is the lion on the far left side of the frame.'
    
    # case 6
    # question = "The man wearing white and blue shirt and baseball cap in the video."
    # image_path = './Q-R1/src/open-r1-multimodal/src/open_r1/videorefer_d_vis_new/37/00160.jpg'
    # image_path = './data/VideoRefer/benchmark/eval/VideoRefer-Bench-D/masked-all-frame-no-countor/1/00008.jpg'
    # answer = "The man is wearing a white and blue striped shirt over a turquoise t-shirt, complemented by a blue baseball cap. He stands in front of a yellow \"Supreme\" banner, making fluid hand gestures as he speaks. His medium complexion is highlighted by a gold necklace with a cross pendant and a gold bracelet on his left wrist, while a small earring adorns his left ear."
    

    if args.brief:
        question = COT_BRIEF_PROMPT.format(question)
        
    if answer is None:
        answer = generate_answer(question, image_path, model, processor)
        print(f"Generated answer: {answer}")
    # exit()
        
    messages = [[
        {"role": "user", "content": [{"type": "image", "image": image_path}, 
        {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to(model.device)
    print(inputs.input_ids)
    # exit()
    # labels = prepare_labels_from_input_ids(inputs.input_ids, im_start_id)
    labels = prepare_labels_from_input_ids(inputs.input_ids, vision_end_id)
    print(f"labels:", labels)
    # exit()
    
    # print(f"input_ids: {inputs.input_ids[0].tolist()}")
    # print(f"labels: {labels[0].tolist()}")
    with torch.inference_mode():
        output = model(**inputs, labels=labels, output_attentions=True, return_dict=True)
    # print(output)
    attentions = output.attentions  # tuple(layers) of list(bsz) of (k_select_len, nheads, q_len)
    # print(f"attentions shape: {attentions[0].shape}")
    
    attentions = [one_layer[0] for one_layer in attentions]  # tuple(layers) of (k_select_len, nheads, q_len)
    attentions = torch.stack(attentions, dim=0)   # (layers, k_select_len, nheads, q_len)
    # print(f"attentions shape: {attentions.shape}")

    # layer_list = [2, 7, 12, 17, 22, 27]  # e.g., [0, 3, 5]
    layer_list = [2, 7, 12, 17, 22, 27]   # e.g., [0, 3, 5]
    attentions = attentions[layer_list]  # (len(layer_list), k_select_len, nheads, q_len)
    attentions = torch.mean(attentions, dim=0)  # (k_select_len, nheads, q_len) —— 已经平均了选定层
    attentions = attentions.mean(dim=1)  # (k_select_len, q_len)

    # attentions = reduce_tensor(attentions, 23, 0)  # (k_select_len, nheads, q_len)
    # attentions = reduce_tensor(attentions, -1, 1)  # (k_select_len, q_len)

    attentions = attentions.transpose(0, 1)  # (q_len, k_select_len)
    attentions = attentions[:-1]
    label_mask = labels != -100
    label_ids = inputs.input_ids[label_mask]  # (q_len, )
    # print(f"attentions shape: {attentions.shape}")

    assert len(label_ids) == attentions.shape[0], f"Label IDs length {len(label_ids)} does not match attentions length {attentions.shape[0]}"
    grid_hw = inputs["image_grid_thw"][0, 1:] // 2
    grid_hw = grid_hw.tolist()
    assert grid_hw[0] * grid_hw[1] == attentions.shape[1], f"Grid size {grid_hw} does not match attentions width {attentions.shape[1]}"
    image_name = os.path.basename(image_path).split('.')[0]
    dir_suffix = f"_layer_23_head{args.head}_new_2"
    if args.brief:
        dir_suffix += "_brief"
    save_dir = os.path.join(args.output_dir,  image_name + dir_suffix)
    os.makedirs(save_dir, exist_ok=True)
    save_attentions_on_image(image_path, attentions, label_ids, grid_hw, save_dir,
                             color_map="jet", alpha=0.5)
        
    id2str_path = os.path.join(save_dir, "id2str.json")
    id2str = {}
    for idx, one_label_id in enumerate(label_ids):
        one_label_str = processor.tokenizer.decode(one_label_id)
        id2str[idx] = (one_label_id.item(), one_label_str)
    with open(id2str_path, 'w') as f:
        json.dump(id2str, f, indent=4, ensure_ascii=False)
    print(f"Saved label IDs to {id2str_path}")
            
        
        
if __name__ == "__main__":
    main()