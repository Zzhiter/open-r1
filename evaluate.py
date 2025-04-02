import json
import os
import re
from vllm import LLM, SamplingParams
import pandas as pd

# 配置参数
# BASE_MODEL_PATH = "Qwen/Qwen2.5-Math-7B-Instruct"
BASE_MODEL_PATH = "/home/dkidna/open-r1/data/Qwen-2.5-7B-Instruct-Simple-RL-epoch-15-2"  # 替换为你的训练好的模型路径
PARQUET_FILE_TRAIN = "/home/dkidna/Search-R1/data6/nq_search/train.parquet"
PARQUET_FILE_TEST = "/home/dkidna/Search-R1/data6/nq_search/test.parquet"
OUTPUT_DIR = "./results"  # 保存测试结果的目录

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print("Loading model...")
llm = LLM(model=BASE_MODEL_PATH)

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,  # 可根据需要调整
    top_p=0.95,
    max_tokens=512  # 根据任务需求调整最大生成长度
)

def load_data_from_parquet(parquet_path):
    """
    从 parquet 文件中加载数据。
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"文件 {parquet_path} 不存在！")
    df = pd.read_parquet(parquet_path)
    return df.to_dict(orient="records")

def extract_answer_from_xml(xml_content):
    """
    从 XML 标签 <answer> 中提取内容。
    :param xml_content: 包含 XML 标签的字符串
    :return: 提取的答案内容（如果未找到则返回 None）
    """
    match = re.search(r"<answer>(.*?)</answer>", xml_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def apply_chat_template(messages, tokenizer):
    """
    使用模型的 Chat Template 格式化输入。
    :param messages: 对话消息列表，例如 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    :param tokenizer: 模型的 Tokenizer
    :return: 格式化后的输入字符串
    """
    chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_template

def generate_predictions(data_samples, tokenizer):
    """
    使用 vllm 模型生成预测。
    """
    results = []

    for sample in data_samples:
        # 构造对话消息
        messages = [
            {"role": "system", "content": sample["prompt"][0]["content"]},
            {"role": "user", "content": sample["prompt"][1]["content"]}
        ]

        # 应用 Chat Template
        formatted_prompt = apply_chat_template(messages, tokenizer)

        # 真实答案
        true_solution = sample["solution"]

        # 生成预测
        outputs = llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        # 提取生成的答案
        extracted_answer = extract_answer_from_xml(generated_text)

        # 记录预测结果
        results.append({
            "prompt": formatted_prompt,
            "true_solution": true_solution,
            "predicted_solution": generated_text,
            "extracted_answer": extracted_answer,
            "is_correct": extracted_answer == true_solution
        })

    return results

def evaluate_model(_class="test"):
    """
    评估模型的效果。
    """
    parquet_file = PARQUET_FILE_TRAIN if _class == "train" else PARQUET_FILE_TEST
    data_samples = load_data_from_parquet(parquet_file)

    # 加载模型的 Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # 生成预测
    results = generate_predictions(data_samples, tokenizer)

    # 统计正确率
    correct_count = sum(result["is_correct"] for result in results)
    total_count = len(results)
    accuracy = correct_count / total_count * 100

    print(f"Evaluation results for {_class}:")
    print(f"Total samples: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    # 保存结果到 JSON 文件
    output_path = os.path.join(OUTPUT_DIR, f"{_class}_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # 测试训练集和测试集
    # evaluate_model(_class="train")
    evaluate_model(_class="test")