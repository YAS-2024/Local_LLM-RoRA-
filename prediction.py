import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import torch

# 推論用データの読み込み
file_path = os.getcwd() + '/data/input/segment/predict_data.xlsx'  # 推論用データのファイルパスを指定
df = pd.read_excel(file_path)
print('!! predict_data loaded successfully !!')

# モデルとトークナイザーのロード
model_path = os.getcwd() + '/model/segment/segment_model'  # 訓練済みモデルの保存先ディレクトリを指定
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
print('!! quantization_config loaded successfully !!')

print("!! Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_path)
print("!! Tokenizer loaded successfully !!")

print("!! Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True  # 低メモリ使用量オプションを明示的に設定
)
model.eval()  # 推論モードに設定
print("!! Model loaded successfully !!")

# デバイスの設定（GPUが利用可能ならGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# バッチ推論用の関数
def predict_batch(texts, model, tokenizer, device, batch_size=2):  # バッチサイズを小さく設定
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        prompts = [f"### extract the sector name from the text.:\n{text}\n\n### answer:\n" for text in batch_texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            # モデルが既に適切なデバイスに配置されているため、inputsをデバイスに移動
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model.generate(inputs['input_ids'], max_length=512 + 50)

        for j, output in enumerate(outputs):
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            split_output = decoded_output.split('### answer:\n')
            if len(split_output) > 1:
                prediction = split_output[1].strip()
            else:
                prediction = ""
            results.append(prediction)
    return results


# 推論を実行
predictions = predict_batch(df['text'].tolist(), model, tokenizer, device)

# 推論結果をデータフレームに追加
df['predicted_sector'] = predictions

# 結果をエクセルファイルに保存
output_file_path = os.getcwd() + '/data/input/segment/predicted_result.xlsx'
df.to_excel(output_file_path, index=False)

print("推論結果が保存されました:", output_file_path)

