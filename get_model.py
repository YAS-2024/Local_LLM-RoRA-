from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
import os
import shutil

# モデルとトークナイザーの名前
model_name = "stabilityai/japanese-stablelm-base-alpha-7b"
tokenizer_name= "novelai/nerdstash-tokenizer-v1"

# 量子化設定の作成
quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # または load_in_4bit=True

# キャッシュディレクトリを削除
cache_dir = "./cache"
shutil.rmtree(cache_dir, ignore_errors=True)

# モデルとトークナイザーのダウンロードと保存
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, additional_special_tokens=['▁▁'])

# モデルのダウンロードと保存
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,        
    trust_remote_code=True
)

# ローカルに保存するディレクトリ
local_dir = os.getcwd() + "/model/stablelm_model"

# 保存
print('モデルの保存')
model.save_pretrained(local_dir)
print('トークナイザー保存')
tokenizer.save_pretrained(local_dir)
print('終わり')


