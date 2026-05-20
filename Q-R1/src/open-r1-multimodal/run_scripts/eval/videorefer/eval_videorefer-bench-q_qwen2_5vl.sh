output_file=qwen2_5vl-videorefer-q-single.json

python3 infer_videorefer_bench_q_qwen2_5vl.py \
--video-folder eval/VideoRefer-Bench-Q/qa-masked-first-frame \
--question-file eval/VideoRefer-Bench-Q/refined-VideoRefer-Bench-Q-synonym.json \
--output-file $output_file 

python eval_scripts/eval_videorefer_bench_q.py \
    --pred-path $output_file

