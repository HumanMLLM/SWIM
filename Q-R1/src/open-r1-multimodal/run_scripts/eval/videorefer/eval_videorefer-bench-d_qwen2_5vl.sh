export HF_ENDPOINT=https://hf-mirror.com
# output_file=qwen2_5vl-sft-videorefer-d-single.json
output_file=qwen2_5vl-sft-videorefer-d-all.json
API_KEY=Your_API_KEY
API_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
# all frame inference
python3 infer_videorefer_bench_d_qwen2_5vl.py \
--video-folder eval/VideoRefer-Bench-D/masked-all-frame \
--question-file eval/VideoRefer-Bench-D/refined-VideoRefer-Bench-D.json \
--output-file $output_file \
--mode all \

# single frame inference
# python3 ../benchmark/infer_videorefer_bench_d_qwen2_5vl.py \
# --video-folder eval/VideoRefer-Bench-D/masked-first-frame \
# --question-file eval/VideoRefer-Bench-D/VideoRefer-Bench-D.json \
# --output-file $output_file \
# --mode single

# gpt_output_file=qwen2_5vl-sft-videorefer-d-gpt-single.json
gpt_output_file=qwen2_5vl-sft-videorefer-d-gpt-all.json

python3 eval_scripts/videorefer_bench_d/1.eval_gpt_new.py \
    --input-file $output_file \
    --output-file $gpt_output_file \
    --api-key $API_KEY \
    --api-endpoint $API_ENDPOINT \
    --api-deployname gpt-4o-2024-11-20

python3 ../videorefer/eval/videorefer_bench_d/2.extract_re.py \
    --input-file $gpt_output_file

python3 ../videorefer/eval/videorefer_bench_d/3.analyze_score.py \
    --input-file $gpt_output_file

echo "qwen25vl"