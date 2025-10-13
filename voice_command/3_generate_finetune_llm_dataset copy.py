import json
import random

# 定義地點與座標
locations = {
    "kitchen": (-4.3018, 3.6791),
    "living room": (6.7558, 2.9908),
    "bedroom": (7.557, -2.0813)
}

# 同義詞庫
synonyms = {
    "go": ["go", "walk to", "move to", "head over to", "proceed to", "step into", "navigate to"],
    "please": ["please", "kindly", "would you", "can you", "I want you to", "make sure to", ""],
    "now": ["now", "right away", "immediately", "asap", "at once", "without delay", ""],
    "the": ["the", "that", "this", ""],
}

# 修飾語
adjectives = ["quickly", "slowly", "carefully", "quietly", "smoothly", "gracefully", ""]

# 語氣隨機模板
templates = [
    "{please} {go} to {the} {place} {now}.",
    "Can you {go} to {the} {place} {now}?",
    "{go} into {the} {place} {now}, {please}.",
    "{please}, {go} {adjective} to {the} {place}.",
    "I need you to {go} to {the} {place} {now}.",
    "{go} {adjective} to {the} {place}.",
    "Make sure to {go} to {the} {place} {now}.",
]

def generate_prompts(place, count=300):
    prompts = []
    for _ in range(count):
        template = random.choice(templates)
        prompt = template.format(
            please=random.choice(synonyms["please"]),
            go=random.choice(synonyms["go"]),
            the=random.choice(synonyms["the"]),
            place=place,
            now=random.choice(synonyms["now"]),
            adjective=random.choice(adjectives),
        )
        # 清理多餘空格
        prompt = " ".join(prompt.split())
        prompts.append(prompt)
    return prompts

# 生成資料集
dataset = []
for place, coords in locations.items():
    prompts = generate_prompts(place, count=300)
    for prompt in prompts:
        dataset.append({
            "prompt": prompt,
            "output": coords
        })

# 存成 JSONL
with open("expanded_dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"已經完成，每個地點 300 筆，共 {len(dataset)} 筆資料。")