#!/usr/bin/env python3
import json
import sys

def main():

    json_path = "./data/VideoRefer/benchmark/eval/VideoRefer-Bench-D/refined-VideoRefer-Bench-D.json"
    index = 23

    try:
        # 读取文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取或解析 JSON 失败: {e}")
        sys.exit(1)

    # 判断数据是数组还是包含数组的对象
    if isinstance(data, dict):
        # 假设数组在 data['data'] 中
        if 'data' in data and isinstance(data['data'], list):
            array_data = data['data']
        else:
            print("JSON 对象中未找到数组字段")
            sys.exit(1)
    elif isinstance(data, list):
        array_data = data
    else:
        print("JSON 格式不支持")
        sys.exit(1)

    # 检查 index 范围
    if index < 0 or index >= len(array_data):
        print(f"索引 {index} 超出范围 (0 ~ {len(array_data)-1})")
        sys.exit(1)

    # 获取 conversions 字段
    item = array_data[index]
    if 'conversations' in item:
        print(item['conversations'])
    else:
        print(f"索引 {index} 中无 'conversations' 字段")

if __name__ == '__main__':
    main()