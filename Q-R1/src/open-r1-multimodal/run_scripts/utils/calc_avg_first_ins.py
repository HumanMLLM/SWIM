import json
import numpy as np

# ==== 读取JSON ====
json_path = "./Q-R1/src/open-r1-multimodal/run_scripts/utils/first_ins_in_ans.json"  # 修改成你的JSON路径
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

per_frame = data["per_frame"]  # dict: {video_id: [ {frame, tokens}, ... ]}

# ==== 获取所有指标名 ====
# 假设所有帧都有同样的tokens结构
first_video = next(iter(per_frame.values()))
first_tokens = first_video[0]["tokens"]
metrics_list = list(first_tokens.keys())

# 去掉 "token" 字段（它是词，不是数值指标）
metrics_list = [m for m in metrics_list if m != "token"]

# ==== 计算 video_avg ====
video_avg = {}
for vid, frame_list in per_frame.items():
    metrics_sum = {metric: [] for metric in metrics_list}
    for frame_info in frame_list:
        tokens_data = frame_info.get("tokens")

        # 跳过空列表或空字典
        if not isinstance(tokens_data, dict) or not tokens_data:
            continue

        for metric in metrics_list:
            metrics_sum[metric].append(tokens_data[metric])

    # 如果该视频所有帧都被跳过，则指标值为 0
    video_avg[vid] = {metric: float(np.mean(vals)) if vals else 0.0
                      for metric, vals in metrics_sum.items()}

# ==== 计算 global_avg ====
global_avg = {metric: float(np.mean([video_avg[v][metric] for v in video_avg]))
              for metric in metrics_list}

# ==== 输出 ====
print("=== Video Average ===")
for vid in sorted(video_avg.keys()):
    print(f"Video {vid}:")
    for metric in metrics_list:
        print(f"  {metric}: {video_avg[vid][metric]:.6f}")
    print()

print("=== Global Average ===")
for metric in metrics_list:
    print(f"{metric}: {global_avg[metric]:.6f}")

# 如果需要保存结果
save_path = "video_global_avg_swim.json"
save_data = {
    "video_avg": video_avg,
    "global_avg": global_avg
}
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
print(f"[INFO] Results saved to {save_path}")