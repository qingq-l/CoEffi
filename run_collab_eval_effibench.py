#This part is the configuration adapted for deepseek-coder, qwen2.5-coder, and opencoder.
import torch
import json
import os
import argparse
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from CustomizedGeneration import DeepSeekModelLM, Qwen2ModelLM, CodeLlamaModelLM

INITIAL_THRESHOLD = 0.5
MIN_THRESHOLD = 0.3
MAX_THRESHOLD = 1.2
WINDOW_SIZE = 10
ADJ_STEP = 0.10
MAIN_TEMP = 0.6
EXPERT_TEMP = 0.8

batch_size = 8

def load_collaborative_model(main_path, expert_path, adapter_path):
    print("\n[1/2] Loading collaborative model...")
    if "qwen" in main_path.lower():
        ModelClass = Qwen2ModelLM
    elif "deepseek" in main_path.lower():
        ModelClass = DeepSeekModelLM
    else:
        ModelClass = CodeLlamaModelLM

    print(f"Loading main model ({ModelClass.__name__})...")
    main_model = ModelClass.from_pretrained(
        main_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(main_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("Loading expert base model...")
    expert_base = AutoModelForCausalLM.from_pretrained(
        expert_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Merging DPO expert weights (LoRA)...")
    expert_model = PeftModel.from_pretrained(expert_base, adapter_path)
    expert_model.eval()

    print("Binding collaborative generation hook...")
    collaborative_generate = partial(
        main_model.generate_with_experts,
        expert_lm=expert_model,
        threshold=INITIAL_THRESHOLD,
        window_size=WINDOW_SIZE,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        adj_step=ADJ_STEP,
        expert_temperature=EXPERT_TEMP,
        expert_top_p=0.95,
        temperature=MAIN_TEMP,
        repetition_penalty=1.0,
        do_sample=True
    )

    main_model.generate = collaborative_generate
    return main_model, tokenizer

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
        print(f"\n[Error] Collaborative generation failed: {e}")
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
    parser = argparse.ArgumentParser(description='Fetch completions using Collaborative Dual-Models.')
    parser.add_argument('--main_model', type=str, default='path/to/main_model')
    parser.add_argument('--expert_base', type=str, default='path/to/expert_base_model')
    parser.add_argument('--expert_adapter', type=str, default='path/to/expert_adapter')
    args = parser.parse_args()

    with open("path/to/dataset.json", "r") as f:
        dataset = json.load(f)

    with open("path/to/prompt.txt", "r") as f:
        prompt_text = f.read()

    model, tokenizer = load_collaborative_model(args.main_model, args.expert_base, args.expert_adapter)

    print("\n[2/2] Running collaborative decoding evaluation on EffiBench...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        dataset[i : i + batch_size] = fetch_completion(
            dataset[i : i + batch_size], model, tokenizer, prompt_text
        )

    save_name = "collab_result.json"
    save_path = os.path.join("results", save_name)
    os.makedirs("results", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"\nDone. Results saved to: {save_path}")



# #The following part is the configuration adapted for StarCoder2.
# import torch
# import json
# import os
# import argparse
# from tqdm import tqdm
# from functools import partial
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# from CustomizedGeneration import ModelWithExperts

# INITIAL_THRESHOLD = 0.5
# MIN_THRESHOLD = 0.3
# MAX_THRESHOLD = 1.2
# WINDOW_SIZE = 10
# ADJ_STEP = 0.10
# MAIN_TEMP = 0.6
# EXPERT_TEMP = 0.8

# batch_size = 8

# def load_collaborative_model(main_path, expert_path, adapter_path):
#     print("\n[1/2] Loading collaborative model (dynamic injection)...")
#     print("Loading main model...")
#     main_model = AutoModelForCausalLM.from_pretrained(
#         main_path,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )

#     main_model.__class__ = type(
#         f"Collab_{main_model.__class__.__name__}",
#         (main_model.__class__, ModelWithExperts),
#         {}
#     )

#     tokenizer = AutoTokenizer.from_pretrained(main_path, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = 'left'

#     print("Loading expert base model...")
#     expert_base = AutoModelForCausalLM.from_pretrained(
#         expert_path,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )

#     print("Merging DPO expert weights (LoRA)...")
#     expert_model = PeftModel.from_pretrained(expert_base, adapter_path)
#     expert_model.eval()

#     print("Binding collaborative generation hook...")
#     collaborative_generate = partial(
#         main_model.generate_with_experts,
#         expert_lm=expert_model,
#         threshold=INITIAL_THRESHOLD,
#         window_size=WINDOW_SIZE,
#         min_threshold=MIN_THRESHOLD,
#         max_threshold=MAX_THRESHOLD,
#         adj_step=ADJ_STEP,
#         expert_temperature=EXPERT_TEMP,
#         expert_top_p=0.95,
#         temperature=MAIN_TEMP,
#         repetition_penalty=1.0,
#         do_sample=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id
#     )

#     main_model.generate = collaborative_generate
#     return main_model, tokenizer

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
#         print(f"\n[Error] Collaborative generation failed: {e}")
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
#     parser = argparse.ArgumentParser(description='Fetch completions using Collaborative Dual-Models.')
#     parser.add_argument('--main_model', type=str, default='path/to/main_model')
#     parser.add_argument('--expert_base', type=str, default='path/to/expert_base_model')
#     parser.add_argument('--expert_adapter', type=str, default='path/to/expert_adapter')
#     args = parser.parse_args()

#     with open("path/to/dataset.json", "r") as f:
#         dataset = json.load(f)

#     with open("path/to/prompt.txt", "r") as f:
#         prompt_text = f.read()

#     model, tokenizer = load_collaborative_model(args.main_model, args.expert_base, args.expert_adapter)

#     print("\n[2/2] Running collaborative decoding evaluation on EffiBench...")
#     for i in tqdm(range(0, len(dataset), batch_size)):
#         dataset[i : i + batch_size] = fetch_completion(
#             dataset[i : i + batch_size], model, tokenizer, prompt_text
#         )

#     save_name = "collab_result.json"
#     save_path = os.path.join("results", save_name)
#     os.makedirs("results", exist_ok=True)
#     with open(save_path, "w") as f:
#         json.dump(dataset, f, indent=4)

#     print(f"\nDone. Results saved to: {save_path}")