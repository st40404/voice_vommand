# Citation of TinyLlama
# @misc{zhang2024tinyllama,
#         title={TinyLlama: An Open-Source Small Language Model},
#         author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
#         year={2024},
#         eprint={2401.02385},
#         archivePrefix={arXiv},
#         primaryClass={cs.CL}
# }

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
import torch
from datasets import load_dataset
from transformers.integrations import TensorBoardCallback
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os
from huggingface_hub import login
from peft import PeftModel

max_seq_length = 2048

# 第一階段訓練完成的模型路徑
first_finetuned_model_path = "./../model/TinyLlama-continue-finetune-chatgpt-prompts"
# first_finetuned_model_path = "TinyLlama/TinyLlama-1.1B-Chat-V0.4"

# 第二階段微調後的模型輸出路徑和 Hugging Face Repository ID
second_finetune_output_path = "./../model/TinyLlama-finetune-navigation"
second_huggingface_repo_id = "st40404/TinyLlama-finetune-navigation"

# ----------------------------------------------------------------------------------------------------
## 1. 載入第一階段微調完成的模型，並重新設定 LoRA
# ----------------------------------------------------------------------------------------------------

print(f"Loading previously fine-tuned model from: {first_finetuned_model_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = first_finetuned_model_path,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
print("Previously fine-tuned model loaded.")

model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                   ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)
print("LoRA adapters re-initialized on the loaded model.")

# ----------------------------------------------------------------------------------------------------
## 2. 準備新的數據集 (fka/awesome-chatgpt-prompts)
# ----------------------------------------------------------------------------------------------------

# print(f"Loading new dataset: fka/awesome-chatgpt-prompts")
# new_dataset_name = "fka/awesome-chatgpt-prompts"
# dataset = load_dataset(new_dataset_name, split="train")

print("Loading new dataset: nav2_point_v3.jsonl")
dataset = load_dataset("json", data_files="./../dataset/nav2_point_v3.jsonl", split="train")


def format_conversation(example):
    user_prompt = example["prompt"]
    assistant_response = example["response"]

    # 這裡我們讓 assistant 的輸出必須包含結束標記
    formatted_text = (
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{assistant_response}\n<|end_of_conversation|>\n"
    )
    # 這邊加入 assistant 後，訓練後的結果都會先輸出 assistant

    # 為了方便訓練，我們只需要提供 user 提示，模型會生成 assistant 的回答
    example["formatted_text"] = formatted_text
    return example

dataset = dataset.map(format_conversation, num_proc=os.cpu_count() or 1) # 使用所有可用的 CPU 核心
# --- 關鍵修改結束 ---

# ----------------------------------------------------------------------------------------------------
## 3. 設定訓練參數與 Trainer
# ----------------------------------------------------------------------------------------------------

class TrainingLoggerCallback(TrainerCallback):
    def __init__(self, log_path=f"./logs/{second_huggingface_repo_id.split('/')[-1]}.txt"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("step\tloss\tlearning_rate\n")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        lr = logs.get("learning_rate")

        with open(self.log_path, "a") as f:
            loss_str = f"{loss:.6f}" if loss is not None else "N/A"
            lr_str = f"{lr:.8f}" if lr is not None else "N/A"
            f.write(f"{step}\t{loss_str}\t{lr_str}\n")


trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="formatted_text",
    max_seq_length=max_seq_length,
    dataset_num_proc=os.cpu_count() or 1, # 使用所有可用的 CPU 核心

    args=UnslothTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        max_steps=8000,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=second_finetune_output_path,
        report_to="tensorboard",
        logging_dir="./logs",
    ),

    callbacks=[TrainingLoggerCallback()],
)

# ----------------------------------------------------------------------------------------------------
## 4. 開始訓練
# ----------------------------------------------------------------------------------------------------
print("Starting second stage fine-tuning...")
trainer_stats = trainer.train()
print("Second stage fine-tuning finished.")

# ----------------------------------------------------------------------------------------------------
## 5. 合併 LoRA 權重並保存完整的微調模型
# ----------------------------------------------------------------------------------------------------

print("Merging LoRA weights back into the model from second stage...")
model.merge_and_unload()
print("LoRA weights merged for second stage.")

print(f"Saving merged model from second stage to {second_finetune_output_path}...")
model.save_pretrained(second_finetune_output_path, safe_serialization=True)
tokenizer.save_pretrained(second_finetune_output_path)
print("Merged model and tokenizer from second stage saved locally.")

# ----------------------------------------------------------------------------------------------------
## 6. (可選) 重新載入合併後的模型進行驗證
# ----------------------------------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"Verifying merged model by reloading from {second_finetune_output_path}...")
reloaded_model = AutoModelForCausalLM.from_pretrained(
    second_finetune_output_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
).to("cuda")
reloaded_tokenizer = AutoTokenizer.from_pretrained(second_finetune_output_path)
print("Merged model from second stage reloaded successfully.")

# ----------------------------------------------------------------------------------------------------
## 7. 將模型和 Tokenizer 上傳到 Hugging Face Hub
# ----------------------------------------------------------------------------------------------------

from huggingface_hub import login

print(f"Pushing merged model from second stage to Hugging Face Hub: {second_huggingface_repo_id}...")
reloaded_model.push_to_hub(second_huggingface_repo_id, private=True)
reloaded_tokenizer.push_to_hub(second_huggingface_repo_id, private=True)
print("Merged model and tokenizer from second stage pushed to Hugging Face Hub successfully!")