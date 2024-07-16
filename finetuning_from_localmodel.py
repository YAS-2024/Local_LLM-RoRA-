import datasets
import peft
from datasets import Dataset
from transformers import AutoModelForCausalLM, LlamaTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments, BitsAndBytesConfig
import pickle
import pandas as pd
import os

# エクセルファイルを読み込み
file_path = os.getcwd() + '/data/train/segment/train_data.xlsx'  # ファイルパスを指定
dr_path = os.getcwd() + '/model/segment/'  # 保存先ディレクトリを指定
temp_path=os.getcwd() + '/temp/'
df = pd.read_excel(file_path)
model_path=os.getcwd() + '/model/stablelm_model'

# 量子化設定の作成
quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # または load_in_4bit=True

# トークナイザーのロード
tokenizer = LlamaTokenizer.from_pretrained(model_path, additional_special_tokens=['▁▁'])
# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    trust_remote_code=True,
)



# 必要な列を抽出
texts = df['text'].astype(str).tolist()  # C列をテキストに変換してリスト化
labels = df['sector'].astype(str).tolist()  # D列をテキストに変換してリスト化

# データフレームを辞書形式に変換
data_dict = {'text': texts, 'sector': labels}

# 辞書をデータセットに変換
dataset = Dataset.from_dict(data_dict)

# データセットをトレーニング用と評価用に分割
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']



# 学習時のプロンプト形式を定義する
prompt_no_context_format = """
### extract the business division name from the text.If there is no business division equivalent, please respond with NONE.If multiple business divisions are extracted, please concatenate them with a hyphen:
{text}

### answer:
{sector}
"""

# データセットの各データをプロンプトに変換してtokenizeをかける
def tokenize(samples):
    prompts = []
    for text, label in zip(samples['text'], samples['sector']):
        prompts.append(prompt_no_context_format.format(text=text, sector=label))
    result = tokenizer(prompts, padding=False, truncation=True, max_length=512)
    return result

# データセットのトークナイズ
tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True)


# 変換したデータセットを保存する
with open(temp_path + "dataset.pickle", "wb") as f:
    pickle.dump(tokenized_train_dataset, f)


# 学習時の必要メモリの軽量化のための設定
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# LoRA用の学習設定
config = peft.LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.01,
    inference_mode=False,
    task_type=peft.TaskType.CAUSAL_LM,
    target_modules=["query_key_value"],
)
model = peft.get_peft_model(model, config)

# 学習を実行
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    args=TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=True,
        save_strategy="no",
        max_steps=1000,  # テストのため少な目
        save_steps=100,
        output_dir="./model",
        report_to="none",
        save_total_limit=3,
        push_to_hub=False,
        auto_find_batch_size=True
    ),
    data_collator=DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()

# LoRAモデルの保存
trainer.model.save_pretrained(dr_path + "segment_model")
tokenizer.save_pretrained(dr_path + "segment_model")
model.config.save_pretrained(dr_path + "segment_model")  # config.jsonの保存