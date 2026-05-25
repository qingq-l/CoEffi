#This part is the configuration adapted for deepseek-coder, qwen2.5-coder, and opencoder.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os
from tqdm import tqdm

batch_size = 8

def construct_prompt_template(inputs, model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    input_tokens = tokenizer.batch_encode_plus(
        inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(model.device)

    try:
        sequences = model.generate(
            inputs=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        input_length = input_tokens.input_ids.shape[1]
        generated_texts = []
        for i in range(len(sequences)):
            new_tokens = sequences[i][input_length:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated_texts.append(decoded)

    except Exception as e:
        print(f"\n[Error] Code generation failed: {e}")
        raise e

    return generated_texts

def fetch_completion(data_entry_lists, model, tokenizer, prompt_text):
    inputs_batchs = []
    for data_entry in data_entry_lists:
        test_case = data_entry["small_test_cases"]
        raw_prompt = (
            f"{prompt_text}\n"
            f"# Task description:\n```python\n{data_entry['markdown_description']}\n```\n"
            f"# Test case:\n```python\n{test_case}\n```"
        )

        messages = [
            {"role": "system", "content": "You are a world-class Python programmer. Please output the code solution directly within a code block."},
            {"role": "user", "content": raw_prompt}
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_batchs.append(chat_text)

    completion_lists = construct_prompt_template(inputs_batchs, model, tokenizer)
    for i in range(len(data_entry_lists)):
        data_entry_lists[i]["completion"] = completion_lists[i]

    return data_entry_lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch completions using huggingface model.')
    parser.add_argument('--model', '-m', type=str, default='path/to/model', help='Model to use for completion')
    args = parser.parse_args()
    model_name = args.model

    with open("path/to/dataset.json", "r") as f:
        dataset = json.load(f)

    with open("path/to/prompt.txt", "r") as f:
        prompt_text = f.read()

    print(f"\nLoading model ({model_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("\nStarting baseline generation...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        dataset[i : i + batch_size] = fetch_completion(
            dataset[i : i + batch_size], model, tokenizer, prompt_text
        )

    save_name = "baseline_result.json"
    save_path = os.path.join("results", save_name)
    os.makedirs("results", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"\nDone. Results saved to: {save_path}")


# #The following part is the configuration adapted for StarCoder2.
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import json
# import argparse
# import os
# from tqdm import tqdm

# batch_size = 8

# def construct_prompt_template(inputs, model, tokenizer):
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = 'left'

#     input_tokens = tokenizer.batch_encode_plus(
#         inputs,
#         padding=True,
#         return_tensors="pt",
#     ).to(model.device)

#     for t in input_tokens:
#         if torch.is_tensor(input_tokens[t]):
#             input_tokens[t] = input_tokens[t].to(model.device)

#     try:
#         sequences = model.generate(
#             inputs=input_tokens["input_ids"],
#             attention_mask=input_tokens["attention_mask"],
#             max_new_tokens=512,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.95,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )

#         input_length = input_tokens.input_ids.shape[1]
#         generated_texts = []
#         for i in range(len(sequences)):
#             new_tokens = sequences[i][input_length:]
#             decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
#             generated_texts.append(decoded)

#     except Exception as e:
#         print(f"\n[Error] Code generation failed: {e}")
#         raise e

#     return generated_texts

# def fetch_completion(data_entry_lists, model, tokenizer, prompt_text):
#     inputs_batchs = []
#     for data_entry in data_entry_lists:
#         test_case = data_entry["small_test_cases"]

#         triple_ticks = chr(96) * 3
#         raw_prompt = (
#             f"{prompt_text}\n"
#             f"# Task description:\n{triple_ticks}python\n{data_entry['markdown_description']}\n{triple_ticks}\n"
#             f"# Test case:\n{triple_ticks}python\n{test_case}\n{triple_ticks}"
#         )

#         chat_text = (
#             f"### Instruction:\n"
#             f"You are a world-class Python programmer. Please output the code solution directly within a code block.\n\n"
#             f"{raw_prompt}\n\n"
#             f"### Response:\n"
#         )
#         inputs_batchs.append(chat_text)

#     completion_lists = construct_prompt_template(inputs_batchs, model, tokenizer)
#     for i in range(len(data_entry_lists)):
#         data_entry_lists[i]["completion"] = completion_lists[i]

#     return data_entry_lists

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Fetch completions using huggingface model.')
#     parser.add_argument('--model', '-m', type=str, default='path/to/model', help='Model to use for completion')
#     args = parser.parse_args()
#     model_name = args.model

#     with open("path/to/dataset.json", "r") as f:
#         dataset = json.load(f)

#     with open("path/to/prompt.txt", "r") as f:
#         prompt_text = f.read()

#     print(f"\nLoading model ({model_name})...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = 'left'

#     print("\nStarting generation...")
#     for i in tqdm(range(0, len(dataset), batch_size)):
#         dataset[i : i + batch_size] = fetch_completion(
#             dataset[i : i + batch_size], model, tokenizer, prompt_text
#         )

#     save_name = "baseline_result.json"
#     save_path = os.path.join("results", save_name)
#     os.makedirs("results", exist_ok=True)
#     with open(save_path, "w") as f:
#         json.dump(dataset, f, indent=4)

#     print(f"\nDone. Results saved to: {save_path}")