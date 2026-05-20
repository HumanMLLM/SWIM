import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import numpy as np
from decord import VideoReader, cpu
import cv2


class VideoRefer_Bench_D_general(Dataset):
    def __init__(self, video_folder, data_list, mode):
        self.video_folder = video_folder
        self.data_list = data_list
        self.mode = mode
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_path = os.path.join(self.video_folder, str(idx))
        video_frames = os.listdir(video_path)
        video_frames = [os.path.join(video_path, vf) for vf in video_frames]
        raw_video = 'VideoRefer-Bench-D/Panda-70M-part/'+line['video']
        vr = VideoReader(raw_video, ctx=cpu(0))
        n = len(vr)
        raw_video_frames = []  # 保存帧图片路径
        raw_video_dir = os.path.join('VideoRefer-Bench-D/VideoRefer-D-raw/', line['video'].replace('.mp4', ''))
        os.makedirs(raw_video_dir, exist_ok=True)

        raw_video_frames = os.listdir(raw_video_dir)
        raw_video_frames = sorted(raw_video_frames)
        raw_video_frames = [os.path.join(raw_video_dir, vf) for vf in raw_video_frames]
        n = len(raw_video_frames)
        if n <= 64:
            raw_video_frames = raw_video_frames  # 不足则全部保留
        else:
            # 生成均匀间隔的索引（这里用 round 强制取整）
            indices = np.linspace(0, n - 1, num=64, dtype=int)
            raw_video_frames =  [raw_video_frames[i] for i in indices]
        if 'conversations' in line.keys():
            return {
                'video': "VideoRefer-Bench-D/Panda-70M-part/"+line['video'],
                'video_frames': video_frames,
                'raw_video_frames': raw_video_frames,
                'question': line['conversations'][0]['value'].replace('<ins>','').replace('</ins>',''),
                'caption': line['conversations'][1]['value'],                
            }
    

def collate_fn(batch):
    video = [x['video'] for x in batch]
    vf = [x['video_frames'] for x in batch]
    qs = [x['question'] for x in batch]
    cap = [x['caption'] for x in batch]
    raw_video_frames = [x['raw_video_frames'] for x in batch]
    return video, vf, qs, cap, raw_video_frames

def build_videorefer_bench_d_eval(args):
    questions = json.load(open(args.question_file))
    dataset = VideoRefer_Bench_D_general(args.video_folder, questions, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    # load model

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "<MODEL_PATH>",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    # load processor
    processor = AutoProcessor.from_pretrained("<MODEL_PATH>")

    val_loader = build_videorefer_bench_d_eval(args)
    
    final_data = []
    ans_file = open(args.output_file, "w")

    
    for i, (videos, video_frames, questions, captions, raw_video_frames) in enumerate(tqdm(val_loader)):
        video = videos[0]
        video_frame = video_frames[0]
        question = questions[0]
        caption = captions[0]
        raw_video_frame = raw_video_frames[0]
        # print(raw_video_frame)
        messages = [
                {
                "role": "system",
                "content": "You are Qwen, a helpful assistant with global scene understanding, capable of grounding user-specified text to the corresponding object instance in the image and providing fine-grained analysis based on visual evidence.",
            },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video, # video, raw_video_frame, video_frame
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
        print(messages)
        # exit()      
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )        
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=9999)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])

        record = {
            'video': video,
            'caption': caption,
            'pred': output_text[0],
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    run_inference(args)
