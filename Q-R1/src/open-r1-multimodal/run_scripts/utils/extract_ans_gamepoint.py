import json
import re

def extract_ins_word(text: str) -> str:
    """提取 <ins>...</ins> 中的内容"""
    match = re.search(r"<ins>(.*?)</ins>", text)
    if match:
        return match.group(1).strip()
    return None

# 读取 JSON 数据
josn_path = './data/DAMO-NLP-SG/VideoRefer-Bench/refined-VideoRefer-Bench-D.json'
with open(josn_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 记录全局出现顺序
global_occurrence_counter = 0   # 计数
results = []  # 保存每条数据结果

# 用于记录某个target_word在全局出现的次数
word_global_counts = {}

metric_json_path = './Q-R1/src/open-r1-multimodal/src/open_r1/attn_vis/all_stat_results.json'
with open(metric_json_path, 'r', encoding='utf-8') as f:
    metric_data = json.load(f)

metric_data_per_frame = metric_data['per_frame']

for idx, one_data_info in enumerate(data):
    question = one_data_info['conversations'][0]['value']
    answer = one_data_info['conversations'][1]['value']
    
    # 提取target_word
    target_word = ' ' + extract_ins_word(question) if extract_ins_word(question) else ''
    
    # 去掉 <ins> 标签
    question = question.replace('<ins>', '').replace('</ins>', '')
    answer = answer.replace('<ins>', '').replace('</ins>', '')
    print(target_word)
    print(question)
    print(answer)
    # print(metric_data_per_frame[str(idx)])
    # exit()
    
    all_pos_in_question = len([m.start() for m in re.finditer(re.escape(target_word), question)])
    print(all_pos_in_question)
    first_pos_in_answer = answer.find(target_word)

    for index_metric, metric_each_frame in enumerate(metric_data_per_frame[str(idx)]):
        if first_pos_in_answer == -1:
            metric_data_per_frame[str(idx)][index_metric]['tokens'] = []
            continue
        metric_data_per_frame[str(idx)][index_metric]['tokens'] = metric_data_per_frame[str(idx)][index_metric]['tokens'][all_pos_in_question]



# 如果需要保存
with open("first_ins_in_ans.json", "w", encoding="utf-8") as f:
    json.dump(metric_data, f, indent=2, ensure_ascii=False)
print("已保存结果到 target_word_global_rank.json")