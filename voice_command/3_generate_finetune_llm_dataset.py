import json
import random

# 定義地點與對應座標
locations = {
    "kitchen": (-4.3018, 3.6791),
    "living room": (6.7558, 2.9908),
    "badroom": (7.557, -2.0813)  # 注意：這裡原始拼字是 badroom
}

# 每個地點要生成的語句樣本
patterns = [
    "go to {place}",
    "move to {place}",
    "navigate to {place}",
    "please go to {place}",
    "go {place}",
    "head to {place}",
    "bring me to {place}",
    "walk to {place}",
    "drive to {place}",
    "go towards {place}",
    "go straight to {place}",
    "head straight to {place}",
    "move towards {place}",
    "reach the {place}",
    "take me to {place}",
    "find the {place}",
    "can you go to {place}",
    "please head to {place}",
    "navigate towards {place}",
    "go in direction of {place}",
    "move in direction of {place}",
    "walk towards {place}",
    "drive towards {place}",
    "move straight to {place}",
    "please take me to {place}",
    "let's go to {place}",
    "go now to {place}",
    "head over to {place}",
    "move over to {place}",
    "get to {place}"
]

# 輸出檔案
with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for place, coords in locations.items():
        for pattern in patterns:
            input_text = pattern.format(place=place)
            data = {
                "prompt": input_text,
                "response": {"x": coords[0], "y": coords[1]}
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("✅ dataset.jsonl 已經生成完成，共 {} 筆資料".format(len(locations) * len(patterns)))