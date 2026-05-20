import os
import json
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
import cv2
from tqdm import tqdm
import shutil
from matplotlib import pyplot as plt
import pycocotools.mask as maskUtils
import re
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLProcessor

COT_BRIEF_PROMPT = "{}\nAnswer the question using a single word or phrase."
torch.set_printoptions(profile="full")
def parse_args():
    parser = argparse.ArgumentParser(description="Save attention weights from Qwen2.5 VL model.")
    parser.add_argument("--vision_info_path", default="./data/DAMO-NLP-SG/VideoRefer-700K/format-videorefer-detailed-caption-125k-part.json", type=str, help="Path to the vision info file.")
    # parser.add_argument("--output_dir",default="./Q-R1/src/open-r1-multimodal/src/open_r1/attn_vis/qwen25vl-videorefer-refined-detailed-125k-detail-125k-insit-21k-miou-multilayer-test-single-turn-2-final-train", type=str, help="Directory to save the attention weights.")
    parser.add_argument("--output_dir",default="./Q-R1/src/open-r1-multimodal/src/open_r1/attn_vis/<MODEL_PATH>-train", type=str, help="Directory to save the attention weights.")
    parser.add_argument("--model_path",default="<MODEL_PATH>", type=str, help="Directory to save the attention weights.")
    # parser.add_argument("--model_path", type=str, default="./Q-R1/src/open-r1-multimodal/output/qwen25vl-videorefer-refined-detailed-125k-detail-125k-insit-21k-miou-multilayer-test-single-turn-2")
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

def annToMask(rle):
    m = maskUtils.decode(rle)
    return m
def reduce_tensor(src: torch.Tensor, select_idx, dim, keepdim=False):
    if select_idx == -1:
        return src.mean(dim=dim, keepdim=keepdim)
    else:
        rtn = src.select(dim, select_idx)
        if keepdim:
            rtn = rtn.unsqueeze(dim)
        return rtn

def save_attentions_on_image(image_path, attentions, label_ids, grid_hw, output_dir, color_map="jet", alpha=0.5):
    
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    grid_h, grid_w = grid_hw
    target_w, target_h = grid_w * 28, grid_h * 28
    image = image.resize((target_w, target_h), Image.LANCZOS)

    for idx, (one_attn, one_label_id) in enumerate(zip(attentions, label_ids)):
        print(idx, one_label_id)
        save_path = os.path.join(output_dir, f"{idx}_{one_label_id.item()}.png")
        attention_img = one_attn.reshape(grid_h, grid_w)
        # minmax norm
        attention_img = (attention_img - attention_img.min()) / (attention_img.max() - attention_img.min())
        # interpolate to image size (nearest neighbor)
        attention_img = attention_img.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_h, grid_w)
        attention_img = torch.nn.functional.interpolate(attention_img, size=(target_h, target_w), mode='nearest')
        attention_img = attention_img.squeeze(0).squeeze(0)  # (height, width)
        attention_img = (attention_img * 255).byte().cpu().numpy()

        heatmap_rgba = cm.get_cmap(color_map)(attention_img)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')
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

def extract_ins_word(question: str) -> str:
    match = re.search(r"<ins>(.*?)</ins>", question)
    if match:
        return match.group(1).strip()
    else:
        return None

def to_serializable(val):
    if isinstance(val, (np.float32, np.float64)):
        return float(val)
    elif isinstance(val, (np.int32, np.int64)):
        return int(val)
    return val
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

    josn_path = './data/DAMO-NLP-SG/VideoRefer-700K/refined-format-videorefer-detailed-caption-0-12k.json'
    # josn_path = './data/DAMO-NLP-SG/VideoRefer-Bench/refined-VideoRefer-Bench-D.json'
    save_root = './Q-R1/src/open-r1-multimodal/src/open_r1/videorefer_d_vis'
    with open(josn_path, 'r') as f:
        vision_info = json.load(f)

    vision_info = vision_info[0:100]
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)
    video_results = {}
    
    for info_idx, one_info in enumerate(vision_info):
        question = one_info['conversations'][0]['value']
        answer = one_info['conversations'][1]['value']
        target_word = ' '+extract_ins_word(question)
        question = question.replace('<ins>','').replace('</ins>', '')
        answer = answer.replace('<ins>','').replace('</ins>', '')

        video_path = os.path.join('./data/DAMO-NLP-SG/VideoRefer-Bench/Panda-70M-part',one_info['video'])
        vreader = VideoReader(video_path)
        num_frames_of_video = len(vreader)
        frame_id_list = frame_sample(num_frames_of_video, video_frames=16)
        video_data = [Image.fromarray(frame) for frame in vreader.get_batch(frame_id_list).asnumpy()]
        save_dir = os.path.join(save_root, str(info_idx))
        os.makedirs(save_dir, exist_ok=True)
        for idx, i in enumerate(frame_id_list):
            image = video_data[idx]
            image.save(os.path.join(save_dir, str(i).zfill(5) + '.jpg'))
            image_path = os.path.join(save_dir, str(i).zfill(5) + '.jpg')
            if ('annotation' not in one_info) or (not isinstance(one_info['annotation'], list)) or (len(one_info['annotation']) == 0):
                print(f"[WARNING] ({index_name}) No annotation found, saving raw frames.")
                continue   
            ann = one_info['annotation'][0]
            if str(i) in ann and ann[str(i)]['segmentation'] is not None:
                mask = annToMask(ann[str(i)]['segmentation'])
            
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
            labels = prepare_labels_from_input_ids(inputs.input_ids, vision_end_id)

            with torch.inference_mode():
                output = model(**inputs, labels=labels, output_attentions=True, return_dict=True)
            attentions = output.attentions  # tuple(layers) of list(bsz) of (k_select_len, nheads, q_len)
            
            attentions = [one_layer[0] for one_layer in attentions]  # tuple(layers) of (k_select_len, nheads, q_len)
            attentions = torch.stack(attentions, dim=0)   # (layers, k_select_len, nheads, q_len)

            layer_list = [2, 7, 12, 17, 22, 27]  # e.g., [0, 3, 5]
            attentions = attentions[layer_list]  # (len(layer_list), k_select_len, nheads, q_len)
            attentions = torch.mean(attentions, dim=0)  # (k_select_len, nheads, q_len) —— 已经平均了选定层
            attentions = attentions.mean(dim=1)  # (k_select_len, q_len)

            # attentions = reduce_tensor(attentions, args.layer, 0)  # (k_select_len, nheads, q_len)
            # attentions = reduce_tensor(attentions, args.head, 1)  # (k_select_len, q_len)

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
            video_name = image_path.split('/')[-2]
            dir_suffix = f"_multi_layer_head{args.head}"
            save_dir_attn = os.path.join(args.output_dir, video_name + '/' + image_name + dir_suffix)
            os.makedirs(save_dir_attn, exist_ok=True)
            save_attentions_on_image(image_path, attentions, label_ids, grid_hw, save_dir_attn,
                                    color_map=args.color_map, alpha=args.alpha)
                
            id2str_path = os.path.join(save_dir_attn, "id2str.json")
            id2str = {}
            for idx, one_label_id in enumerate(label_ids):
                one_label_str = processor.tokenizer.decode(one_label_id)
                id2str[idx] = (one_label_id.item(), one_label_str)
            with open(id2str_path, 'w') as f:
                json.dump(id2str, f, indent=4, ensure_ascii=False)
            print(f"Saved label IDs to {id2str_path}")

            # Step 1: 找 target word 每个 token attention map（不取 mean）
            target_token_ids = processor.tokenizer.encode(target_word, add_special_tokens=False)
            token_results = []  # 保存每个 token 的指标

            for token_pos in [ti for ti, tid in enumerate(label_ids) if tid.item() in target_token_ids]:
                token_str = processor.tokenizer.decode(label_ids[token_pos])

                # token 对应的 cross attention (k_len,)
                token_attn_map = attentions[token_pos]

                # 注意力归一化
                token_attn_img = token_attn_map.reshape(grid_hw[0], grid_hw[1])
                token_attn_img = (token_attn_img - token_attn_img.min()) / (token_attn_img.max() - token_attn_img.min() + 1e-8)

                # 上采样到原图大小
                token_attn_img_up = torch.nn.functional.interpolate(
                    token_attn_img.unsqueeze(0).unsqueeze(0),
                    size=mask.shape[:2],
                    mode='nearest'
                ).squeeze().cpu().float().numpy()

                # 阈值筛选高亮区域
                threshold = 0.5 # 用命令行参数
                highlight = token_attn_img_up > threshold
                mask_bool = mask.astype(bool)

                # 计算 TP / FP / FN / TN
                TP = float(np.sum(highlight & mask_bool))
                FP = float(np.sum(highlight & ~mask_bool))
                FN = float(np.sum(~highlight & mask_bool))
                TN = float(np.sum(~highlight & ~mask_bool))

                # IOU
                iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

                # Dice & F1
                dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
                f1 = dice  # 在二分类分割中 F1 等于 Dice

                # Precision
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

                # Recall
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                # Specificity
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

                # Pearson 相关系数（保护零方差情况）
                if np.std(highlight) == 0 or np.std(mask_bool) == 0:
                    pearson = 0.0
                else:
                    pearson = float(np.corrcoef(
                        highlight.flatten().astype(float),
                        mask_bool.flatten().astype(float)
                    )[0, 1])

                # MCC（注意类型和零除保护）
                denom_val = float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                denom = np.sqrt(denom_val) if denom_val > 0 else 0.0
                mcc = ((TP * TN) - (FP * FN)) / denom if denom > 0 else 0.0

                # 原 overlap_score: mask 内高亮 / 高亮总数
                highlight_total = float(np.sum(highlight))
                overlap_score = TP / highlight_total if highlight_total > 0 else 0.0

                # GamePoint@1
                flat_attn = token_attn_img_up.flatten()
                sorted_idx = np.argsort(flat_attn)[::-1]  # 降序
                k_count_1 = min(1, len(flat_attn))
                topk_idx_1 = sorted_idx[:k_count_1]
                gp_k_1 = np.sum(mask_bool.flatten()[topk_idx_1]) / k_count_1 if k_count_1 > 0 else 0.0

                # GamePoint@5
                flat_attn = token_attn_img_up.flatten()
                sorted_idx = np.argsort(flat_attn)[::-1]  # 降序
                k_count_5 = min(5, len(flat_attn))
                topk_idx_5 = sorted_idx[:k_count_5]
                gp_k_5 = np.sum(mask_bool.flatten()[topk_idx_5]) / k_count_5 if k_count_5 > 0 else 0.0
            
                # GamePoint@10
                flat_attn = token_attn_img_up.flatten()
                sorted_idx = np.argsort(flat_attn)[::-1]  # 降序
                k_count_10 = min(10, len(flat_attn))
                topk_idx_10 = sorted_idx[:k_count_10]
                gp_k_10 = np.sum(mask_bool.flatten()[topk_idx_10]) / k_count_10 if k_count_10 > 0 else 0.0

                # GamePoint@50
                flat_attn = token_attn_img_up.flatten()
                sorted_idx = np.argsort(flat_attn)[::-1]  # 降序
                k_count_50 = min(50, len(flat_attn))
                topk_idx_50 = sorted_idx[:k_count_50]
                gp_k_50 = np.sum(mask_bool.flatten()[topk_idx_50]) / k_count_50 if k_count_50 > 0 else 0.0

                # GamePoint@100
                flat_attn = token_attn_img_up.flatten()
                sorted_idx = np.argsort(flat_attn)[::-1]  # 降序
                k_count_100 = min(100, len(flat_attn))
                topk_idx_100 = sorted_idx[:k_count_100]
                gp_k_100 = np.sum(mask_bool.flatten()[topk_idx_100]) / k_count_100 if k_count_100 > 0 else 0.0

                # GamePoint@0.01
                p_count_1 = int(len(flat_attn) * 0.01)
                top_p_idx_1 = sorted_idx[:p_count_1]
                gp_p_1 = np.sum(mask_bool.flatten()[top_p_idx_1]) / p_count_1 if p_count_1 > 0 else 0.0

                # GamePoint@0.05
                p_count_5 = int(len(flat_attn) * 0.05)
                top_p_idx_5 = sorted_idx[:p_count_5]
                gp_p_5 = np.sum(mask_bool.flatten()[top_p_idx_5]) / p_count_5 if p_count_5 > 0 else 0.0
            
                # GamePoint@0.1
                p_count_10 = int(len(flat_attn) * 0.1)
                top_p_idx_10 = sorted_idx[:p_count_10]
                gp_p_10 = np.sum(mask_bool.flatten()[top_p_idx_10]) / p_count_10 if p_count_10 > 0 else 0.0


                # 额外离散 / 排序相关性指标
                y_true = mask_bool.flatten().astype(int)
                y_score = flat_attn
                spearman_corr, _ = spearmanr(y_score, y_true)
                kendall_corr, _ = kendalltau(y_score, y_true)
                ap_score = average_precision_score(y_true, y_score) if np.any(y_true) else 0.0
                try:
                    auc_score = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
                except ValueError:
                    auc_score = 0.0

                norm_map = (token_attn_img_up - token_attn_img_up.mean()) / (token_attn_img_up.std() + 1e-8)
                nss_score = norm_map[mask_bool].mean() if np.any(mask_bool) else 0.0

                p_attn = token_attn_img_up / (token_attn_img_up.sum() + 1e-8)
                if mask_bool.sum() > 0:
                    p_mask = mask_bool.astype(float) / mask_bool.sum()
                    kl_div = np.sum(p_mask * np.log((p_mask + 1e-8) / (p_attn + 1e-8)))
                else:
                    kl_div = 0.0

                token_results.append({
                    "token": token_str,
                    "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                    "overlap": overlap_score,
                    "iou": iou, "dice": dice, "f1": f1,
                    "precision": precision, "recall": recall, "specificity": specificity,
                    "pearson": pearson, "mcc": mcc,
                    "gamepoint_k_1": gp_k_1, 
                    "gamepoint_k_5": gp_k_5, 
                    "gamepoint_k_10": gp_k_10, 
                    "gamepoint_k_50": gp_k_50, 
                    "gamepoint_k_100": gp_k_100, 
                    "gamepoint_p_1": gp_p_1,
                    "gamepoint_p_5": gp_p_5,
                    "gamepoint_p_10": gp_p_10,
                    "spearman": spearman_corr,
                    "kendall": kendall_corr,
                    "ap": ap_score,
                    "auc": auc_score,
                    "nss": nss_score,
                    "kl_div": kl_div,
                })

            # Step 3: 保存到 video_results
            video_results.setdefault(video_name, []).append({
                "frame": image_name,
                "tokens": token_results
            })
    # 全局统计（video_avg 汇总所有指标平均值 + global_avg 汇总所有视频平均值）
    video_avg = {}
    # 为方便 global_avg 计算，提前定义指标列表（和 token_info 中的 key 对应）
    metrics_list = [
        "TP",
        "FP",
        "FN",
        "TN",
        "overlap",
        "iou",
        "dice",
        "f1",
        "precision",
        "recall",
        "specificity",
        "pearson",
        "mcc",
        "gamepoint_k_1",
        "gamepoint_k_5",
        "gamepoint_k_10",
        "gamepoint_k_50",
        "gamepoint_k_100",
        "gamepoint_p_1",
        "gamepoint_p_5",
        "gamepoint_p_10",
        "spearman",
        "kendall",
        "ap",
        "auc",
        "nss",
        "kl_div",
    ]

    for vn, frames in video_results.items():
        # 收集该视频所有 token 的指标
        metrics_sum = {metric: [] for metric in metrics_list}
        for f in frames:
            for token_info in f["tokens"]:
                for metric_name in metrics_list:
                    metrics_sum[metric_name].append(token_info[metric_name])

        # 计算该视频的平均值
        video_avg[vn] = {
            metric_name: float(np.mean(vals)) if vals else 0.0
            for metric_name, vals in metrics_sum.items()
        }

    # 计算全部视频的平均指标（global_avg）
    global_avg = {
        metric: float(np.mean([v_metrics[metric] for v_metrics in video_avg.values()])) if video_avg else 0.0
        for metric in metrics_list
    }

    # 保存成 JSON
    save_json = {
        "per_frame": video_results,  # 每帧详细数据
        "video_avg": video_avg,      # 每视频平均指标
        "global_avg": global_avg     # 全局平均指标
    }
    save_path = os.path.join(args.output_dir, "all_stat_results.json")
    with open(save_path, "w") as f:
        json.dump(save_json, f, indent=4, default=to_serializable, ensure_ascii=False)

    print(f"[INFO] Saved all statistics to {save_path}")
    print("[INFO] Global mean of all videos:", global_avg)

            
        
        
if __name__ == "__main__":
    main()