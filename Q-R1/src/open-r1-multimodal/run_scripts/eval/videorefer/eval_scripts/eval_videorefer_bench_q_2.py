import json
import re

# ======= 在这里直接设置文件路径 =======
pred_path = "./data/VideoRefer/benchmark/qwen2_5vl-videorefer-q-single.json"        # 输入预测数据文件（每行一个JSON对象或JSON数组均可）
error_save_path = "SWIM_benchmark_q_errors.json"       # 输出错误信息文件
# ====================================

def load_jsonlines(path):
    """读取 JSON Lines 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    # 统计结果
    all_sum = {}
    right_num = {}
    errors = []

    data = load_jsonlines(pred_path)

    for i, d in enumerate(data):
        # 提取正确答案字母
        gt = d['Answer']
        match = re.search(r'\(([A-Z])\)', gt)
        if match:
            gt_letter = match.group(1)
        else:
            gt_letter = gt.strip()[0]

        # 提取预测答案字母
        pred = d['pred']
        match = re.search(r'\(([A-Z])\)', pred)
        if match:
            pred_letter = match.group(1)
        else:
            match = re.search(r'([A-Z])\)', pred)
            if match:
                pred_letter = match.group(1)
            else:
                pred_letter = pred.strip().replace('.', '')[0]

        # 初始化统计
        if d['type'] not in all_sum:
            all_sum[d['type']] = 0
            right_num[d['type']] = 0

        # 判断是否正确
        if pred_letter.lower() == gt_letter.lower():
            right_num[d['type']] += 1
        else:
            errors.append({
                "index": i,
                "video": d['video'],
                "type": d['type'],
                "Answer_raw": d['Answer'],
                "Answer_letter": gt_letter,
                "Pred_raw": d['pred'],
                "Pred_letter": pred_letter
            })

        all_sum[d['type']] += 1

    # 输出每种类型的结果
    all_total, all_right = 0, 0
    for t in all_sum:
        acc = right_num[t] / all_sum[t] if all_sum[t] else 0
        print(f"####### {t} #######")
        print(f"all num  : {all_sum[t]}")
        print(f"right num: {right_num[t]}")
        print(f"accuracy : {acc:.4f}")
        all_total += all_sum[t]
        all_right += right_num[t]

    # 输出总结果
    acc_total = all_right / all_total if all_total else 0
    print("####### average #######")
    print(f"all num  : {all_total}")
    print(f"right num: {all_right}")
    print(f"accuracy : {acc_total:.4f}")

    # 保存错误信息
    with open(error_save_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"❌ 错误样本已保存到 {error_save_path}，共 {len(errors)} 条")

if __name__ == "__main__":
    main()