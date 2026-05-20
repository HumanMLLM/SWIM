# VLM-R1: A stable and generalizable R1-style Large Vision-Language Model

<font size=4><div align='center' > [[🤗 REC Demo](https://huggingface.co/spaces/omlab/VLM-R1-Referral-Expression)] [[🤗 OVD Demo](https://huggingface.co/spaces/omlab/VLM-R1-OVD)] [[🤗 REC Data](https://huggingface.co/datasets/omlab/VLM-R1)] [[🤗 Checkpoints](https://huggingface.co/collections/omlab/vlm-r1-models-67b7352db15c19d57157c348)] [[📝 Blog](https://om-ai-lab.github.io/index.html)] </div></font>


<div align="center">
<img src="./assets/performance4.png" width="900"/>
<div>
  <font size=4>
    <p>🎉  <b>Our VLM-R1 Math model reaches the top of the Open-Compass Math Leaderboard (under 4B parameters) and OVD model achieves the state-of-the-art performance on OVDEval.</b></p>
  </font>
</div>
</div>



Since the introduction of [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1), numerous works have emerged focusing on reproducing and improving upon it. In this project, we propose VLM-R1, a stable and generalizable R1-style Large Vision-Language Model. 

Specifically, for the task of Referring Expression Comprehension (REC), we trained [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) using both R1 and SFT approaches. The results reveal that, on the in-domain test data, the performance of the SFT model shows little change compared to that of the R1 model base model when the number of training steps is relatively small (100–600 steps), while the R1 model shows a steady improvement (as shown at the left of the figure below). More importantly, on the out-of-domain test data, the SFT model’s performance deteriorates slightly as the number of steps increases. Nevertheless, the RL model generalizes its reasoning ability to the out-of-domain data (as shown at the right of the figure below).

![image](./assets/performance3.png)
\* *We found previous REC SFT exps used a mismatch pixel config. Therefore, we re-run the study with the correct config on a more complex out-of-domain data. See our [findings](https://om-ai-lab.github.io/2025_03_24.html) for details.*

## 🚀 Features
This repository supports:
- **`Full Fine-tuning for GRPO`**: see [run_grpo_rec.sh](src/open-r1-multimodal/run_scripts/run_grpo_rec.sh)
- **`Freeze Vision Modules`**: set `freeze_vision_modules` as `true` in the script.
- **`LoRA Fine-tuning for GRPO`**: see [run_grpo_rec_lora.sh](src/open-r1-multimodal/run_scripts/run_grpo_rec_lora.sh)
- **`Multi-node Training`**: see [multinode_training_demo.sh](src/open-r1-multimodal/run_scripts/multinode_training_demo.sh)
- **`Multi-image Input Training`**: see [run_grpo_gui.sh](src/open-r1-multimodal/run_scripts/run_grpo_gui.sh)
- **`For your own data`**: see [here](#for-your-own-data)
- **`Support various VLMs`**: see [How to add a new model](assets/add_new_model.md), now we support QwenVL and InternVL


## 🗞️ Update
- **`2025-03-23`**: 🔥 We release the VLM-R1-OVD [model weights](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321) and [demo](https://huggingface.co/spaces/omlab/VLM-R1-OVD), which shows the state-of-the-art performance on OVDEval. Welcome to use it.
- **`2025-03-20`**: 🔥 We achieved SOTA results on [OVDEval](https://github.com/om-ai-lab/OVDEval) with our RL-based model, outperforming SFT baselines and specialized object detection models. Read our [blog post](https://om-ai-lab.github.io/2025_03_20.html) for details on how reinforcement learning enhances object detection performance.
- **`2025-03-17`**: Our VLM-R1 Math model reaches the top of the [Open-Compass Math Leaderboard](https://rank.opencompass.org.cn/leaderboard-multimodal-reasoning/?m=REALTIME) (under 4B parameters). We have released the [checkpoint](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-Math-0305).
- **`2025-03-15`**: We support multi-image input data. Check the format of multi-image input [here](#for-your-own-data). We also provide an example of multi-image script [run_grpo_gui.sh](src/open-r1-multimodal/run_scripts/run_grpo_gui.sh), see [here](#for-your-own-data) for details.
- **`2025-03-13`**: We support InternVL for GRPO. See [run_grpo_rec_internvl.sh](src/open-r1-multimodal/run_scripts/run_grpo_rec_internvl.sh) for details. The annotation json files used in InternVL are [here](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_internvl.zip). If you want to add your new model, please refer to [How to add a new model](assets/add_new_model.md).
- **`2025-03-02`**: We support LoRA Fine-tuning for GRPO. See [run_grpo_rec_lora.sh](src/open-r1-multimodal/run_scripts/run_grpo_rec_lora.sh) for details.
- **`2025-02-27`**: We support the `number of iterations per batch` and `epsilon value for clipping` in the original GRPO algorithm with args: `--num_iterations` and `--epsilon`.
- **`2025-02-25`**: We support multi-node training for GRPO. See [multinode_training_demo.sh](src/open-r1-multimodal/run_scripts/multinode_training_demo.sh) for details.
- **`2025-02-21`**: We release the [checkpoint](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps) of the VLM-R1 REC model.
- **`2025-02-20`**: We release the script for [general data loading](#for-your-own-data).
- **`2025-02-19`**: We incorporate an explanation of the [SFT](#sft) method.
- **`2025-02-17`**: We release the VLM-R1 REC [Demo](https://huggingface.co/spaces/omlab/VLM-R1-Referral-Expression) on Hugging Face Spaces.
- **`2025-02-15`**: We release the VLM-R1 repository and [GRPO](#grpo) training script.


## 🤖 Models
- **[`OVD`](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321)**: Trained with VLM-R1, our Open-Vocabulary Detection (OVD) model achieves the state-of-the-art performance on OVDEval.
- **[`Math`](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-Math-0305)**: Through VLM-R1 training, our math model focuses on multimodal reasoning tasks and has achieved Top1 on the OpenCompass Multi-modal Reasoning Leaderboard among models < 4B.
- **[`REC`](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps)**: Trained with VLM-R1, our Referring Expression Comprehension (REC) model showcases the superior performance on out-of-domain data and a series of reasoning-grounding tasks.

| Version | Base VLM | Checkpoint | Task Type | 
|----------|----------------|---------------|----------------|
| VLM-R1-Qwen2.5VL-3B-OVD-0321 | Qwen2.5VL-3B | [omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321) | Open-Vocabulary Detection | 
| VLM-R1-Qwen2.5VL-3B-Math-0305 | Qwen2.5VL-3B | [omlab/VLM-R1-Qwen2.5VL-3B-Math-0305](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-Math-0305) | Multi-Modal Math | 
| VLM-R1-Qwen2.5VL-3B-REC-500steps | Qwen2.5VL-3B | [omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps) | REC/Reasoning-Grounding | 


## 🎯 ToDo
- [x] Implement multi-node training.
- [x] Implement LoRA Fine-tuning.
- [x] Support more Multimodal LLMs.
- [x] Support multi-image input.
- [x] Release the VLM-R1 Math model.
- [x] Release the blog of VLM-R1.
- [x] Release the VLM-R1-OVD model.
- [ ] Release the technical report of VLM-R1.
- [ ] Study cross task generalization.
- [ ] Enhance VLM for other tasks [welcome issue]. 



## 🛠️ Setup

```bash
conda create -n vlm-r1 python=3.10
conda activate vlm-r1
bash setup.sh
```

## 💪🏻 Training

### Referring Expression Comprehension (REC)

#### 📚 GRPO

1. Download the [COCO Train2014 image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip) and unzip it, and we refer to the image dir as `<your_image_root>`.

2. Download the [RefCOCO/+/g and LISA-Grounding Annotation files](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip) and unzip it (LISA-Grounding is used for out-of-domain evaluation).

3. Write the path of the annotation files in the `src/open-r1-multimodal/data_config/rec.yaml` file.
```bash
datasets:
    - json_path: /path/to/refcoco_train.json
    - json_path: /path/to/refcocop_train.json
    - json_path: /path/to/refcocog_train.json
```

4. ```bash src/open-r1-multimodal/run_scripts/run_grpo_rec.sh```

> [!NOTE] 
> If you encounter 'CUDA out of memory' error, you can try to (1) set `gradient_checkpointing` as `true`, (2) reduce the `per_device_train_batch_size`, or (3) use lora.

```bash
cd src/open-r1-multimodal

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/rec.yaml \
    --image_root <your_image_root> \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --freeze_vision_modules false # If you want to only finetune the language model, set this to true.
```
<div align="center">
<img src="./assets/iou.jpg" width="750"/>
</div>
<!-- ![image](./assets/wandb.jpg) -->

#### 📚 Multi-Node GRPO
For multi-node training, please refers to [multinode_training_demo.sh](src/open-r1-multimodal/multinode_training_demo.sh).

#### 📚 SFT
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train the SFT model.
1. Clone the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository and install the dependencies.
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

2. Download the dataset_info.json, mllm_rec_json.json, and qwen2_5_vl_full_sft.yaml we provided [here](https://huggingface.co/datasets/omlab/VLM-R1/tree/main/sft_related). Put the json files in the `LLaMA-Factory/data` directory and the yaml file in the `LLaMA-Factory/examples/train_full` directory.

3. Run the following command to train the SFT model.
```bash
llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft.yaml
```

### For your own data

<div style="text-align: justify;">

We also support data loading the jsonl data of this format in [`src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`](src/open-r1-multimodal/src/open_r1/grpo_jsonl.py). Please note that you may need to use different reward functions for your specialized tasks. Welcome to PR to add your own reward functions or share any other interesting findings!

</div>

The jsonl has the format as follows:
```json
{
  "id": 1,
  "image": "Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16885.png",
  "conversations": [
    {"from": "human", "value": "<image>What number of purple metallic balls are there?"},
    {"from": "gpt", "value": "0"}
  ]
}
```
If you want to use multi-image input, you can use the following format:
```json
{
  "id": 1,
  "image": ["Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16885.png", "Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16886.png"],
  "conversations": [
    {"from": "human", "value": "<image><image>What number of purple metallic balls in total within the two images?"},
    {"from": "gpt", "value": "3"}
  ]
}
```
> [!NOTE] 
> The image path in the jsonl file should be relative to the image folder specified in `--image_folders`. The absolute path of the input image is constructed as `os.path.join(image_folder, data['image'])`. For example:
- If your jsonl has `"image": "folder1/image1.jpg"`
- And you specify `--image_folders "/path/to/images/"`
- The full image path will be `/path/to/images/folder1/image1.jpg`

Multiple data files and image folders can be specified using ":" as a separator:
```bash
--data_file_paths /path/to/data1.jsonl:/path/to/data2.jsonl \
--image_folders /path/to/images1/:/path/to/images2/
```

The script can be run like this:
```bash
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
  src/open_r1/grpo_jsonl.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero3.json \
    --dataset_name <your_dataset_name> \
    --data_file_paths /path/to/your/data.jsonl \ # can be multiple, separated by ":"
    --image_folders /path/to/your/image/folder/ \ # can be multiple, separated by ":"
    ...
```

<div style="text-align: justify;">

We provide an example of multi-image script [run_grpo_gui.sh](src/open-r1-multimodal/run_scripts/run_grpo_gui.sh). This task requires the model to analyze two GUI screenshots, taken before and after a user action, to determine if any UI interaction defects are present, which is from [GUI-Testing-Arena](https://huggingface.co/datasets/songjah/GTArena-UI-Defects). Download the [image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/gui_multi-image.zip) and unzip it into the `/path/to/images/`. Then modify the `image_folders` parameter in the script and run it.
```bash
bash src/open-r1-multimodal/run_scripts/run_grpo_gui.sh
```

</div>

## 📊 Evaluation

![image](./assets/data2.png)

1. Download the provided [LISA-Grounding images](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/lisa-test.zip).
```bash
cd ./src/eval

# Remember to change the model path, image root, and annotation path in the script
torchrun --nproc_per_node="X" test_rec_r1.py # for GRPO. 'X' is the number of GPUs you have.
torchrun --nproc_per_node="X" test_rec_baseline.py # for SFT.
```

## 🤝 Acknowledgements

We would like to express our sincere gratitude to [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [RefCOCO](https://github.com/lichengunc/refer), [RefGTA](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [OVDEval](https://github.com/om-ai-lab/OVDEval), [GUI-Testing-Arena](https://huggingface.co/datasets/songjah/GTArena-UI-Defects), and [LISA](https://github.com/dvlab-research/LISA) for providing open-source resources that contributed to the development of this project.



## ⭐️ Citation
If you find this project useful, welcome to cite us.
```bib
@misc{shen2025vlmr1,
  author       = {Shen, Haozhan and Zhang, Zilun and Zhao, Kangjia and Zhang, Qianqian and Xu, Ruochen and Zhao, Tiancheng},
  title        = {VLM-R1: A stable and generalizable R1-style Large Vision-Language Model},
  howpublished = {\url{https://github.com/om-ai-lab/VLM-R1}},
  note         = {Accessed: 2025-02-15},
  year         = {2025}
}
```
