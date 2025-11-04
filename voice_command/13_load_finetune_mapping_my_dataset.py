# mapping_test_v5.py (優化後的測試程式)
# ==========================================
# TinyLlama mapping 微調模型測試（優化格式 V5）
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
# 1️⃣ 基本設定
# ------------------------------------------
N_SAMPLES = 50  # 測試樣本數
TEMP_JSON_PATH = "mapping_test_v5.jsonl"
OUTPUT_CSV_PATH = "mapping_test_v5_result.csv"
WRONG_CSV_PATH = "mapping_test_v5_result_wrong.csv"
MODEL_PATH = "./TinyLlama-finetune-mapping"  # 你訓練好的模型資料夾
NOTFOUND_RATIO = 0.15
# 統一為訓練時設定的標準負面回應
NEGATIVE_RESPONSE_STANDARD = "no such coordinates" 
COORD_REGEX = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
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
    "coordinates of {place}?",
    "give me the coordinates of {place}.",
    "find {place}.",
    "tell me the coordinates for {place}."
]
FORMATS = [
    "{loc}: ({x},{y})",
    "{loc} at ({x}, {y})",
    "{loc} = {x},{y}",
    "{loc}: [{x}, {y}]",
    "{loc} → ({x},{y})",
]
# ------------------------------------------
# 2️⃣ 生成測試資料（與訓練資料生成邏輯 V2 一致）
# ------------------------------------------
def generate_mapping_test_dataset(n_samples=N_SAMPLES, filename=TEMP_JSON_PATH):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in tqdm(range(n_samples), desc="Generating mapping test dataset (v5 format)"):
            num_locs = random.randint(3, 8)
            chosen_locs = random.sample(LOCATIONS, k=num_locs)
            coords = {loc: (random.randint(-50, 50), random.randint(-50, 50)) for loc in chosen_locs}
            # 80% 機率含負號
            for loc in coords:
                if random.random() < 0.8:
                    x_sign = random.choice([-1, 1])
                    y_sign = random.choice([-1, 1])
                    x, y = coords[loc]
                    coords[loc] = (x * x_sign, y * y_sign)
            # 15% 機率問不到的地點
            ask_notfound = random.random() < NOTFOUND_RATIO
            if ask_notfound:
                available = [loc for loc in LOCATIONS if loc not in chosen_locs]
                query_loc = random.choice(available)
            else:
                query_loc = random.choice(chosen_locs)
            query_sentence = random.choice(QUERY_TEMPLATES).format(place=query_loc)
            # 打亂 mapping 順序
            random.shuffle(chosen_locs)
            coord_map = {loc: coords[loc] for loc in chosen_locs}
            # 多樣化格式
            lines = []
            for loc in chosen_locs:
                fmt = random.choice(FORMATS)
                x, y = coord_map[loc]
                lines.append(fmt.format(loc=loc, x=x, y=y))
            mapping_text = "\n".join(lines)
            input_text = f"{mapping_text}\n\nQuestion: {query_sentence}"
            
            # 💡 統一輸出 "no such coordinates"
            if ask_notfound:
                output_text = NEGATIVE_RESPONSE_STANDARD 
            else:
                x, y = coord_map[query_loc]
                output_text = f"({x},{y})"
                
            json.dump({"input": input_text, "output": output_text}, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ 已生成 {n_samples} 筆測試資料至 {filename}")
# ------------------------------------------
# 3️⃣ 載入模型
# ------------------------------------------
print(f"🚀 載入模型中：{MODEL_PATH} ...")
# 載入邏輯不變
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
print("✅ 模型載入完成。")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model.to(device)
except Exception:
    pass
# ------------------------------------------
# 4️⃣ 評估函式 (關鍵優化)
# ------------------------------------------
def extract_first_coordinate(text: str):
    # 提取第一個座標
    if text is None:
        return None
    m = COORD_REGEX.search(text)
    if not m:
        return None
    x, y = m.group(1), m.group(2)
    return f"({x},{y})"

def evaluate_model_on_dataset(dataset_path=TEMP_JSON_PATH, output_csv=OUTPUT_CSV_PATH, wrong_csv=WRONG_CSV_PATH):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f]
    results = []
    correct = 0
    for data in tqdm(data_list, desc="Evaluating model"):
        user_prompt = data["input"]
        # 統一清理正確答案
        correct_output = data["output"].strip().lower().replace(" ", "").replace("'","").replace('"',"") 
        
        system_prompt = (
            "Extract coordinates. "
            "Output ONLY (x,y) if found. "
            "Output ONLY 'no such coordinates' if not found. "
            "No extra text."
        )
        
        # 🚨 關鍵優化 1: 使用與訓練一致的 Prompt 格式
        # 這裡使用 train.py 中建議的 Llama/Mistral/TinyLlama 標準指令格式
        # prompt = (
        #     f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] "
        # )

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1] # 取得輸入長度

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
                eos_token_id=[
                    tokenizer.eos_token_id, # 確保用 tokenizer.eos_token_id (可能為 </s>)
                    tokenizer.convert_tokens_to_ids("[/INST]"), # 有時模型會重複 [INST]
                ],
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 🚨 關鍵優化 2: 僅解碼模型生成的部分
        generated_tokens = outputs[0][input_length:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
        
        # 🚨 關鍵優化 3: 嚴格比對輸出
        pred_coord = extract_first_coordinate(decoded)
        
        if pred_coord:
            pred_clean = pred_coord.replace(" ", "")
        elif NEGATIVE_RESPONSE_STANDARD in decoded:
            pred_clean = NEGATIVE_RESPONSE_STANDARD
        # 統一輸出後，不再需要比對 "location not found" 和 "unknown place"
        else:
            # 如果模型輸出既非座標也非標準錯誤訊息，則視為錯誤，並記錄原始輸出
            pred_clean = decoded.split('\n')[0].strip() 
            
        is_correct = (pred_clean == correct_output)
        
        if is_correct:
            correct += 1
        results.append({
            "input": prompt,
            "expected": correct_output,
            "predicted_raw": decoded,
            "predicted_clean": pred_clean,
            "correct": is_correct,
        })
        
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n📊 共 {total} 筆，正確 {correct} 筆，準確率 = {accuracy:.2f}%")
    # 輸出結果
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["input", "expected", "predicted_raw", "predicted_clean", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"✅ 已輸出完整結果至 {output_csv}")
    wrongs = [r for r in results if not r["correct"]]
    if wrongs:
        with open(wrong_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in wrongs:
                writer.writerow(r)
        print(f"❌ 錯誤樣本已輸出至 {wrong_csv}，共 {len(wrongs)} 筆。")
    else:
        print("🎉 無錯誤樣本。")
    return accuracy, results
# ------------------------------------------
# 5️⃣ 主流程
# ------------------------------------------
if __name__ == "__main__":
    generate_mapping_test_dataset()
    evaluate_model_on_dataset()