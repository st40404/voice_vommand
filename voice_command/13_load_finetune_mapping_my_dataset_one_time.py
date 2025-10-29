# Citation of TinyLlama
# @misc{zhang2024tinyllama,
#       title={TinyLlama: An Open-Source Small Language Model}, 
#       author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
#       year={2024},
#       eprint={2401.02385},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL}
# }

from unsloth import FastLanguageModel

# 載入你已經微調好的模型，注意這裡的路徑要正確
# 如果你已經上傳到 Hugging Face Hub，可以從那裡下載
# 例如: model_name = "st40404/TinyLlama-finetune-hermes-end-conversation-gemini"
model_name = "./TinyLlama-finetune-mapping"

# 下載 & 載入模型 (用 Unsloth 最佳化版本)
print(f"載入模型：{model_name}...")
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_name,
#     max_seq_length = 2048,
#     dtype = None,
#     load_in_4bit = True,
# )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,   # base 模型
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-V0.4",   # base 模型
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 再載入 adapter
# model.load_adapter(model_name)

from transformers import TextStreamer

user_prompt = """bedroom locate at (0,4)
balcony locate at (5,3)
hallway locate at (2,7)
office locate at (9,-5)
pantry locate at (4,4)
QUERY: what is the location of pantry?
ANSWER FORMAT: (x,y)"""

prompt = (
    f"<|im_start|>system\nYou are a mapping assistant. Always answer only with coordinates in (x,y) format.<|im_end|>\n" # ICL 提示作為用戶輸入
    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)




print("------------------------------------")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 設定 streamer 可以即時輸出文字
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

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
    streamer=streamer
)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=500,
#     # do_sample=False,goog
#     do_sample=True,
#     temperature=0.8,
#     top_k=50,
#     top_p=0.8,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.pad_token_id,
#     streamer=streamer
# )


# 參數介紹
"""
max_new_tokens : 最多產生幾個新 token (不包含 prompt)
do_sample : 啟用「隨機採樣」，讓回答有變化, 啟用時配合 temperature 與 top_p 使用
temperature : 控制隨機性, 越小越保守、越大越創意, 一般介於 0.3 ~ 1.2
top_k : 全名 Top-K Sampling, 只從機率前 K 高的 token 中挑選, 通常設為 40~100, 太小會侷限回答
top_p : Top-P Sampling (又稱 nucleus sampling), 使用 nucleus 採樣，僅考慮累積機率達到 p 的 token 集合。值越小，生成越保守
eos_token_id : 結束 token, 讓生成提前結束 通常保留 tokenizer.eos_token_id
pad_token_id : padding 用, 在 batch 推論中對齊 (必填)
streamer : 實時顯示輸出, 若不需要串流，可以註解掉
"""


print("------------------------------------")
print ("prompt : " + tokenizer.decode(inputs["input_ids"][0]))
