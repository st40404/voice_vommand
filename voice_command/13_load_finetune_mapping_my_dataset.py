# ==========================================
# Mapping 自動生成 + 模型測試 + 準確率統計 + 匯出 CSV
# ==========================================

import json
import random
import csv
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# ------------------------------------------
# 1️⃣ 參數設定
# ------------------------------------------

N_SAMPLES = 500  # 測試樣本數
TEMP_JSON_PATH = "mapping_test_500.jsonl"
OUTPUT_CSV_PATH = "mapping_test_results.csv"
MODEL_PATH = "./TinyLlama-finetune-mapping"

LOCATIONS = [
    "kitchen", "bedroom", "bathroom", "living room", "garage", "office",
    "garden", "balcony", "dining room", "hallway", "attic", "basement",
    "rooftop", "study room", "storage", "laundry room", "guest room",
    "playroom", "conference room", "server room", "gym", "library",
    "theater", "parking lot", "workshop"
]

QUERY_TEMPLATES = [
    "where is {place}?",
    "what is the location of {place}?",
    "coordinates of {place}?"
]


# ------------------------------------------
# 2️⃣ 生成測試數據集 (500 筆)
# ------------------------------------------

def generate_mapping_test_dataset(n_samples=N_SAMPLES, filename=TEMP_JSON_PATH):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n_samples), desc="Generating mapping test dataset"):
            num_locs = random.randint(3, 8)
            chosen_locs = random.sample(LOCATIONS, k=num_locs)
            coords = {loc: (random.randint(-20, 20), random.randint(-20, 20)) for loc in chosen_locs}

            position_weights = [0.1 if i < num_locs//2 else 0.2 for i in range(num_locs)]
            query_idx = random.choices(range(num_locs), weights=position_weights)[0]
            query_loc = chosen_locs[query_idx]
            query_sentence = random.choice(QUERY_TEMPLATES).format(place=query_loc)

            random.shuffle(chosen_locs)
            coord_map = {loc: coords[loc] for loc in chosen_locs}

            input_lines = [f"{loc} locate at ({coord_map[loc][0]},{coord_map[loc][1]})"
                           for loc in chosen_locs]
            input_lines.append(query_sentence)

            input_text = "\n".join(input_lines)
            output_text = f"({coord_map[query_loc][0]},{coord_map[query_loc][1]})"

            json.dump({"input": input_text, "output": output_text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 已生成 {n_samples} 筆測試資料至 {filename}")


# ------------------------------------------
# 3️⃣ 載入模型
# ------------------------------------------

print(f"🚀 載入模型中：{MODEL_PATH} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
print("✅ 模型載入完成。")


# ------------------------------------------
# 4️⃣ 自動測試 + 計算準確率 + 匯出 CSV
# ------------------------------------------

def evaluate_model_on_dataset(dataset_path=TEMP_JSON_PATH, output_csv=OUTPUT_CSV_PATH):
    correct = 0
    total = 0
    results = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f]

    for data in tqdm(data_list, desc="Evaluating model"):
        user_prompt = data["input"]
        correct_output = data["output"].strip()

        # 包裝成 prompt 格式
        prompt = f"""<|im_start|>system
You are a mapping assistant. Always answer only with coordinates in (x,y) format.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.1,
            top_k=30,
            top_p=0.9,
            eos_token_id=[
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|end_of_conversation|>")],
            pad_token_id=tokenizer.pad_token_id,
        )



        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction.split("assistant")[-1].strip()

        # 評估正確與否
        is_correct = correct_output in prediction
        total += 1
        correct += int(is_correct)

        results.append({
            "input": user_prompt,
            "expected": correct_output,
            "predicted": prediction,
            "correct": is_correct
        })

    accuracy = correct / total * 100
    print(f"\n📊 總共 {total} 筆，正確 {correct} 筆，準確率 = {accuracy:.2f}%")

    # 匯出 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["input", "expected", "predicted", "correct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 結果已輸出到 {output_csv}")
    return accuracy


# ------------------------------------------
# 5️⃣ 主流程
# ------------------------------------------

if __name__ == "__main__":
    generate_mapping_test_dataset()
    evaluate_model_on_dataset()