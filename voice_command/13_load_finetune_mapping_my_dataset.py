# mapping_test_v3.py
# ==========================================
# Mapping 自動生成 + TinyLlama 模型測試 + 精確座標比對 + 匯出 CSV (含錯誤樣本)
# ==========================================

import json
import random
import csv
import re
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch
import os

# ------------------------------------------
# 1️⃣ 參數設定
# ------------------------------------------

N_SAMPLES = 30  # 測試樣本數
TEMP_JSON_PATH = "mapping_test_100.jsonl"
OUTPUT_CSV_PATH = "mapping_test_result.csv"
WRONG_CSV_PATH = "mapping_test_result_wrong.csv"
MODEL_PATH = "./TinyLlama-finetune-mapping"  # 你訓練完的模型資料夾

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

COORD_REGEX = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")

# ------------------------------------------
# 2️⃣ 生成測試數據集 (同訓練格式)
# ------------------------------------------

def generate_mapping_test_dataset(n_samples=N_SAMPLES, filename=TEMP_JSON_PATH):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n_samples), desc="Generating mapping test dataset (v2 format)"):
            num_locs = random.randint(3, 10)
            chosen_locs = random.sample(LOCATIONS, k=num_locs)
            coords = {loc: (random.randint(-50, 50), random.randint(-50, 50)) for loc in chosen_locs}

            # 約 70% 負座標機率
            for loc in coords:
                if random.random() < 0.7:
                    coords[loc] = (coords[loc][0] * random.choice([-1, 1]),
                                   coords[loc][1] * random.choice([-1, 1]))

            query_idx = random.choices(range(num_locs), weights=[0.05] + [0.15] * (num_locs - 1))[0]
            query_loc = chosen_locs[query_idx]
            query_sentence = random.choice(QUERY_TEMPLATES).format(place=query_loc)

            random.shuffle(chosen_locs)
            coord_map = {loc: coords[loc] for loc in chosen_locs}

            input_lines = []
            for loc in chosen_locs:
                x, y = coord_map[loc]
                input_lines.append(f"{loc}: ({x},{y})")
            input_lines.append(f"QUERY: {query_sentence}")
            # 更口語化的格式約束（跟訓練時一致）
            input_lines.append("ANSWER FORMAT: (x,y)")

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

# 把 model 放到 cuda（若支援）
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model.to(device)
except Exception:
    pass  # FastLanguageModel 可能已內部處理 device

# ------------------------------------------
# 工具函式：精確抽座標
# ------------------------------------------

def extract_first_coordinate(text: str):
    """
    從文字中抽出第一個 (x,y)，回傳像 '(x,y)' 的字串，或 None。
    """
    if text is None:
        return None
    m = COORD_REGEX.search(text)
    if not m:
        return None
    x, y = m.group(1), m.group(2)
    return f"({x},{y})"

# ------------------------------------------
# 4️⃣ 自動測試 + 精確比對 + 匯出 CSV（含錯誤樣本）
# ------------------------------------------

def evaluate_model_on_dataset(dataset_path=TEMP_JSON_PATH, output_csv=OUTPUT_CSV_PATH, wrong_csv=WRONG_CSV_PATH):
    results = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f]

    total = len(data_list)
    correct = 0

    for data in tqdm(data_list, desc="Evaluating model"):
        user_prompt = data["input"]
        correct_output = data["output"].strip().replace(" ", "")

        # 與訓練一致的 prompt
        prompt = f"""<|im_start|>system
You are a mapping assistant. Always answer only with coordinates in (x,y) format.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # deterministic for mapping
                temperature=0.1,
                top_k=30,
                top_p=0.9,
                eos_token_id=[
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    tokenizer.convert_tokens_to_ids("<|end_of_conversation|>"),
                ],
                pad_token_id=tokenizer.pad_token_id,
            )

        # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_extracted = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 取得 assistant 回答段落
        # if "<|im_start|>assistant" in decoded:
        #     decoded = decoded.split("<|im_start|>assistant")[-1].strip()
        # # 移除可能的結尾 token 與後續文字
        # decoded = decoded.split("<|end_of_conversation|>")[0].strip()

        # 抽出第一組 (x,y)
        # predicted_extracted = extract_first_coordinate(decoded)
        is_correct = (predicted_extracted is not None) and (predicted_extracted.replace(" ", "") == correct_output)

        if is_correct:
            correct += 1

        results.append({
            "input": user_prompt,
            "expected": correct_output,
            # "predicted_raw": decoded,
            "predicted_extracted": predicted_extracted,
            "correct": is_correct
        })

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n📊 總共 {total} 筆，正確 {correct} 筆，準確率 = {accuracy:.2f}%")

    # 匯出全部結果 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["input", "expected", "predicted_raw", "predicted_extracted", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"✅ 全部結果已輸出到 {output_csv}")

    # 匯出錯誤樣本 CSV（方便人工檢查）
    wrongs = [r for r in results if not r["correct"]]
    if wrongs:
        with open(wrong_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["input", "expected", "predicted_raw", "predicted_extracted", "correct"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in wrongs:
                writer.writerow(r)
        print(f"❌ 錯誤樣本已輸出到 {wrong_csv} （共 {len(wrongs)} 筆）")
    else:
        print("🎉 無錯誤樣本。")

    return accuracy, results

# ------------------------------------------
# 5️⃣ 主流程
# ------------------------------------------

if __name__ == "__main__":
    # 1) 生成測試集
    generate_mapping_test_dataset()

    # 2) 執行評估並輸出 CSV
    evaluate_model_on_dataset()
