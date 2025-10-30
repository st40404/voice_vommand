import json
import random
from tqdm import tqdm

# =============== 可調參數 ===============
N_SAMPLES = 50000
OUTPUT_PATH = "11_mapping_25_6_50000.jsonl"
NOTFOUND_RATIO = 0.15  # 15% 問題問到不存在的地點

NEGATIVE_RESPONSES = [
    "no such coordinates",
    "location not found",
    "unknown place",
]
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

def generate_mapping_dataset():
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _ in tqdm(range(N_SAMPLES), desc="Generating mapping_v3_with_notfound"):
            num_locs = random.randint(3, 8)
            chosen_locs = random.sample(LOCATIONS, k=num_locs)
            coords = {loc: (random.randint(-50, 50), random.randint(-50, 50)) for loc in chosen_locs}

            # 80% 機率讓座標有負號
            for loc in coords:
                if random.random() < 0.8:
                    x_sign = random.choice([-1, 1])
                    y_sign = random.choice([-1, 1])
                    x, y = coords[loc]
                    coords[loc] = (x * x_sign, y * y_sign)

            # ============================
            # 新增「問不到的地點」邏輯
            # ============================
            ask_notfound = random.random() < NOTFOUND_RATIO
            if ask_notfound:
                # 問一個完全沒出現在 mapping 的地點
                available = [loc for loc in LOCATIONS if loc not in chosen_locs]
                query_loc = random.choice(available)
            else:
                # 問 mapping 中的地點
                query_idx = random.choices(range(num_locs), weights=[0.05] + [0.2]*(num_locs-1))[0]
                query_loc = chosen_locs[query_idx]

            query_sentence = random.choice(QUERY_TEMPLATES).format(place=query_loc)

            # 隨機打亂順序
            random.shuffle(chosen_locs)
            coord_map = {loc: coords[loc] for loc in chosen_locs}

            # 多樣化格式
            formats = [
                "{loc}: ({x},{y})",
                "{loc} at ({x}, {y})",
                "{loc} = {x},{y}",
                "{loc}: [{x}, {y}]",
                "{loc} → ({x},{y})",
            ]
            lines = []
            for loc in chosen_locs:
                fmt = random.choice(formats)
                x, y = coord_map[loc]
                lines.append(fmt.format(loc=loc, x=x, y=y))

            # 20% 加入錯誤示範（僅在 input）
            if random.random() < 0.2 and len(chosen_locs) > 1:
                wrong_target = random.choice(chosen_locs)
                wrong_source = random.choice([l for l in chosen_locs if l != wrong_target])
                x, y = coord_map[wrong_source]
                lines.append(f"WRONG: {wrong_target}: ({x},{y})  # ignore this line")

            # 10% 加入雜訊行
            if random.random() < 0.1:
                lines.insert(random.randint(0, len(lines)), "noise: irrelevant data")

            mapping_text = "\n".join(lines)
            input_text = f"{mapping_text}\n\nQuestion: {query_sentence}"

            # ============================
            # 根據查詢是否存在決定輸出
            # ============================
            if ask_notfound:
                output_text = random.choice(NEGATIVE_RESPONSES)
            else:
                x, y = coord_map[query_loc]
                output_text = f"({x},{y})"

            sample = {
                "system": (
                    "Extract coordinates. "
                    "Output ONLY (x,y) if found. "
                    "Output ONLY 'no such coordinates' if not found. "
                    "No extra text."),
                "input": input_text,
                "output": output_text
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ Generated {N_SAMPLES} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_mapping_dataset()