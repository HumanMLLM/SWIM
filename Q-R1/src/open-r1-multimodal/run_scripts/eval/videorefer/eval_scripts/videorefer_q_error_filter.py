import json
from collections import defaultdict

# ========= 配置 =========
file1_path = "./data/VideoRefer/videorefer/eval/SWIM_benchmark_q_errors.json"
file2_path = "./data/VideoRefer/videorefer/eval/qwen2_5_benchmark_q_errors.json"
output_path = "diff_by_type.json"
# =======================

def load_json(path):
    """读取标准JSON文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 读取两个 JSON 文件
    json1 = load_json(file1_path)
    json2 = load_json(file2_path)

    # 用 index 做集合（你也可以改成其它唯一标识）
    json1_ids = {item["index"] for item in json1}

    # 找出在 json2 中有，但 json1 中没有的元素
    diff_items = [item for item in json2 if item["index"] not in json1_ids]

    # 按 type 归类
    grouped = defaultdict(list)
    for item in diff_items:
        grouped[item["type"]].append(item)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"✅ 已将差异数据按 type 分组保存到 {output_path}")
    print(f"共 {len(diff_items)} 条缺失数据，涉及 {len(grouped)} 个 type")

if __name__ == "__main__":
    main()
