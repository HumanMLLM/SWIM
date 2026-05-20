import os
import json
import argparse
import torch
from PIL import Image
import numpy as np
import re
from tqdm import tqdm
import pycocotools.mask as maskUtils
from decord import VideoReader
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# --------- utils ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Compute attention-mask overlap for <ins> words.")
    parser.add_argument("--dataset_json",
                        default="./data/DAMO-NLP-SG/VideoRefer-700K/refined-format-videorefer-detailed-caption-125k-part.json")
    parser.add_argument("--video_root", 
                        default="./data/DAMO-NLP-SG/VideoRefer-Bench/Panda-70M-part")
    parser.add_argument("--output_dir",
                        default="./Q-R1/src/open-r1-multimodal/src/open_r1/attn_vis/qwen25vl-videorefer-refined-detailed-125k-detail-125k-insit-21k-miou-multilayer-test-single-turn-2-test")
    parser.add_argument("--model_path",
                        default="./Q-R1/src/open-r1-multimodal/output/qwen25vl-videorefer-refined-detailed-125k-detail-125k-insit-21k-miou-multilayer-test-single-turn-2")
    parser.add_argument("--layer_list", type=int, nargs="+", default=[2, 7, 12, 17, 22, 27],
                        help="Layers to average attention from")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Attention highlight threshold")
    parser.add_argument("--save_vis", action="store_true",
                        help="Whether to save attention highlight visualizations")
    return parser.parse_args()

def annToMask(rle):
    return maskUtils.decode(rle)

def prepare_labels_from_input_ids(input_ids, im_start_id):
    B, L = input_ids.shape
    labels = input_ids.clone()
    mask = input_ids == im_start_id
    flipped_mask = mask.flip(dims=(1,))
    first_idx_in_flipped = torch.argmax(flipped_mask.int(), dim=1)
    last_pos = (L - 1) - first_idx_in_flipped
    mask_until_idx = last_pos + 3
    mask_until_idx = torch.clamp(mask_until_idx, max=L)
    arange_l = torch.arange(L, device=input_ids.device).expand(B, -1)
    modification_mask = arange_l < mask_until_idx.unsqueeze(1)
    labels[modification_mask] = -100
    return labels

def extract_all_ins_words(question: str):
    """返回所有被<ins></ins>包裹的词组成的列表"""
    matches = re.findall(r"<ins>(.*?)</ins>", question)
    return [m.strip() for m in matches]

def frame_sample(duration, video_frames=16):
    num_frames = 32
    seg_size = float(duration - 1) / num_frames
    raw_frame_ids = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        raw_frame_ids.append((start + end) // 2)
    sampled_frame_ids = []
    seg_size = float(num_frames - 1) / video_frames
    for i in range(video_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        sampled_frame_ids.append(raw_frame_ids[(start + end) // 2])
    return sampled_frame_ids

def compute_overlap(attn_map_np, mask_np, threshold=0.5):
    attn_highlight = attn_map_np > threshold
    highlight_pixels = attn_highlight.sum()
    if highlight_pixels > 0:
        intersection_pixels = np.logical_and(attn_highlight, mask_np).sum()
        return intersection_pixels / highlight_pixels
    return 0.0

# -------- main ----------
def main():
    args = parse_args()

    # load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    vision_end_id = processor.tokenizer.encode("<|vision_end|>")[0]

    with open(args.dataset_json, "r") as f:
        vision_info = json.load(f)

    results = {}
    vision_info = vision_info[0:2]
    for one_info in tqdm(vision_info):
        video_path = os.path.join(args.video_root, one_info['video'])
        vreader = VideoReader(video_path)
        num_frames = len(vreader)
        frame_ids = frame_sample(num_frames, video_frames=16)
        frames = [Image.fromarray(frame) for frame in vreader.get_batch(frame_ids).asnumpy()]

        question_raw = one_info['conversations'][0]['value']
        answer_raw = one_info['conversations'][1]['value']
        ins_words = extract_all_ins_words(question_raw)
        # remove <ins> tags for feeding to model
        question = question_raw.replace('<ins>', '').replace('</ins>', '')
        answer = answer_raw.replace('<ins>', '').replace('</ins>', '')

        results.setdefault(one_info['index'], {})

        for idx_frame, frame_id in enumerate(frame_ids):
            frame_img = frames[idx_frame]
            image_path = os.path.join(args.output_dir, "tmp.jpg")
            os.makedirs(args.output_dir, exist_ok=True)
            frame_img.save(image_path)

            # load mask if exists
            mask = None
            if 'annotation' in one_info and isinstance(one_info['annotation'], list) and len(one_info['annotation']) > 0:
                ann = one_info['annotation'][0]
                if str(frame_id) in ann and ann[str(frame_id)]['segmentation'] is not None:
                    mask = annToMask(ann[str(frame_id)]['segmentation'])

            messages = [
                [
                    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]}
                ]
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            labels = prepare_labels_from_input_ids(inputs.input_ids, vision_end_id)

            with torch.inference_mode():
                output = model(**inputs, labels=labels, output_attentions=True, return_dict=True)

            # attention: (layers, k_select_len, nheads, q_len)
            attentions = [layer[0] for layer in output.attentions]  
            attentions = torch.stack(attentions, dim=0)
            attentions = attentions[args.layer_list].mean(dim=0).mean(dim=1)  # (k_select_len, q_len)
            attentions = attentions.transpose(0, 1)[:-1]  # (q_len, k_select_len)

            label_mask = labels != -100
            label_ids = inputs.input_ids[label_mask]

            grid_hw = (inputs["image_grid_thw"][0, 1:] // 2).tolist()  # (grid_h, grid_w)
            if mask is not None:
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize((grid_hw[1], grid_hw[0]), Image.NEAREST)
                mask_bin = np.array(mask_img) > 0
            else:
                mask_bin = None

            # loop over all ins words
            frame_results = {}
            for word in ins_words:
                # tokenize target word
                target_token_ids = processor.tokenizer.encode(word, add_special_tokens=False)

                # find continuous match in label_ids
                target_positions = []
                for i in range(len(label_ids) - len(target_token_ids) + 1):
                    if label_ids[i:i+len(target_token_ids)].tolist() == target_token_ids:
                        target_positions = list(range(i, i + len(target_token_ids)))
                        break
                if not target_positions:
                    frame_results[word] = None
                    continue
                print(f"{word}: {target_positions}")

                target_attn = attentions[target_positions].mean(dim=0)  # target token avg attention
                attn_map = target_attn.reshape(grid_hw)
                attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
                attn_np = attn_norm.cpu().numpy()

                if mask_bin is not None:
                    overlap = compute_overlap(attn_np, mask_bin, threshold=args.threshold)
                else:
                    overlap = None
                frame_results[word] = overlap

                if True:
                    # 可视化保存
                    vis_img = (attn_norm.unsqueeze(0).unsqueeze(0) * 255).byte()
                    vis_img = torch.nn.functional.interpolate(vis_img, size=(grid_hw[0]*28, grid_hw[1]*28), mode='nearest')
                    vis_np = vis_img.squeeze().cpu().numpy()
                    from matplotlib import cm
                    heatmap_rgba = cm.get_cmap("jet")(vis_np)
                    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
                    blended = Image.blend(frame_img.convert("RGB"), Image.fromarray(heatmap_rgb), alpha=0.5)
                    save_vis_dir = os.path.join(args.output_dir, one_info['index'], word)
                    os.makedirs(save_vis_dir, exist_ok=True)
                    print(f"[INFO] Saving visualization to {save_vis_dir}")
                    blended.save(os.path.join(save_vis_dir, f"{frame_id:05d}.png"))

            results[one_info['index']][f"frame_{frame_id:05d}"] = frame_results

    # save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "ins_words_overlap.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved results to {out_json}")

if __name__ == "__main__":
    main()