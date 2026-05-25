#This part is the configuration adapted for deepseek-coder, qwen2.5-coder, and opencoder.
import os
import sys
import json
import torch
import gc
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from CustomizedGeneration import Qwen2ModelLM, CodeLlamaModelLM, DeepSeekModelLM
except ImportError:
    print("Error: Cannot import CustomizedGeneration. Ensure it is in the same directory.")
    sys.exit(1)

MAIN_MODEL_PATH = "path/to/main_model"
EXPERT_BASE_PATH = "path/to/expert_base_model"
EXPERT_ADAPTER_PATH = "path/to/expert_adapter"

INITIAL_THRESHOLD = 0.5
MIN_THRESHOLD = 0.3
MAX_THRESHOLD = 1.2
WINDOW_SIZE = 10
ADJ_STEP = 0.10
MAIN_TEMP = 0.6
EXPERT_TEMP = 0.8
NUM_SAMPLES = 1

def load_collaborative_model():
    print("\n[1/3] Assembling collaborative model (adapter class version)...")
    
    if "qwen" in MAIN_MODEL_PATH.lower():
        ModelClass = Qwen2ModelLM
    elif "deepseek" in MAIN_MODEL_PATH.lower():
        ModelClass = DeepSeekModelLM
    else:
        ModelClass = CodeLlamaModelLM

    print(f"Loading main model ({ModelClass.__name__})...")
    main_model = ModelClass.from_pretrained(
        MAIN_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("Loading expert base model...")
    expert_base = AutoModelForCausalLM.from_pretrained(
        EXPERT_BASE_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    
    print("Merging DPO expert weights (LoRA)...")
    expert_model = PeftModel.from_pretrained(expert_base, EXPERT_ADAPTER_PATH)
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
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True
    )
    
    main_model.generate = collaborative_generate
    return main_model, tokenizer

def generate_collaborative_completions(model, tokenizer, prompt, n_sample=1):
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Please provide the complete, runnable Python code to solve the problem."},
        {"role": "user", "content": f"Please provide the complete Python solution. Include the imports, function signature, and full implementation. Output your code inside a ```python block.\n\n```python\n{prompt}\n```"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text] * n_sample, return_tensors="pt").to("cuda")

    generate_ids = model.generate(
        inputs=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = model_inputs.input_ids.shape[1]
    completions = []
    
    for i in range(len(generate_ids)):
        new_tokens = generate_ids[i][input_length:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        final_code = decoded
        if "```python" in decoded:
            try:
                final_code = decoded.split("```python")[1].split("```")[0]
            except:
                pass
        elif "```" in decoded:
            try:
                final_code = decoded.split("```")[1].split("```")[0]
            except:
                pass
            
        final_code = final_code.rstrip().lstrip('\n')
        
        if "def " in final_code:
            full_code = final_code
        else:
            full_code = prompt + "\n" + final_code
            
        completions.append(full_code)
    return completions

def main():
    collab_model, collab_tokenizer = load_collaborative_model()

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "collab_result.json")
    
    prompts_file = "path/to/prompts.jsonl"
    print(f"\n[2/3] Reading prompt file {prompts_file}...")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    enamel_results = {}
    print("\n[3/3] Starting collaborative generation...")
    
    first_print_done = False
    
    for line in tqdm(lines, desc="Collaborative Generation"):
        data = json.loads(line.strip())
        task_id_str = data["task_id"]
        prompt_text = data["prompt"]
        
        problem_idx = task_id_str.split("/")[-1]
        
        completions = generate_collaborative_completions(collab_model, collab_tokenizer, prompt_text, NUM_SAMPLES)
        enamel_results[str(problem_idx)] = completions
        
        if not first_print_done:
            print(f"\n[Preview] First task ({task_id_str}) generated:\n{enamel_results[str(problem_idx)][0][:300]}...\n")
            first_print_done = True

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(enamel_results, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*60)
    print(f"Done. Results saved to: {save_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()


# #The following part is the configuration adapted for StarCoder2.
# import os
# import sys
# import json
# import torch
# import gc
# from functools import partial
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# try:
#     from CustomizedGeneration import ModelWithExperts
# except ImportError:
#     print("Error: Cannot import CustomizedGeneration. Ensure it is in the same directory.")
#     sys.exit(1)

# MAIN_MODEL_PATH = "path/to/main_model"
# EXPERT_BASE_PATH = "path/to/expert_base_model"
# EXPERT_ADAPTER_PATH = "path/to/expert_adapter"

# INITIAL_THRESHOLD = 0.5
# MIN_THRESHOLD = 0.3
# MAX_THRESHOLD = 1.2
# WINDOW_SIZE = 10
# ADJ_STEP = 0.10
# MAIN_TEMP = 0.6
# EXPERT_TEMP = 0.8
# NUM_SAMPLES = 1

# def load_collaborative_model():
#     print("\n[1/3] Assembling collaborative model (dynamic injection version)...")

#     print("Loading main model...")
#     main_model = AutoModelForCausalLM.from_pretrained(
#         MAIN_MODEL_PATH,
#         torch_dtype=torch.float16,
#         device_map="cuda:0",
#         trust_remote_code=True
#     )
    
#     main_model.__class__ = type(
#         f"Collab_{main_model.__class__.__name__}",
#         (main_model.__class__, ModelWithExperts),
#         {}
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_PATH, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = 'left'

#     print("Loading expert base model...")
#     expert_base = AutoModelForCausalLM.from_pretrained(
#         EXPERT_BASE_PATH,
#         torch_dtype=torch.float16,
#         device_map="cuda:0",
#         trust_remote_code=True
#     )
    
#     print("Merging DPO expert weights (LoRA)...")
#     expert_model = PeftModel.from_pretrained(expert_base, EXPERT_ADAPTER_PATH)
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
#         top_p=0.95,
#         repetition_penalty=1.0,
#         do_sample=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id
#     )
    
#     main_model.generate = collaborative_generate
#     return main_model, tokenizer

# def generate_collaborative_completions(model, tokenizer, prompt, n_sample=1):
#     enhanced_prompt = (
#         "You are an expert Python programmer. Please provide the complete, runnable Python code to solve the problem. "
#         "Include the imports, function signature, and full implementation. Output your code inside a ```python block.\n\n"
#         f"```python\n{prompt}\n```"
#     )
    
#     chat_text = (
#         "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n"
#         f"{enhanced_prompt}\n\n"
#         "### Response:\n"
#     )
    
#     model_inputs = tokenizer([chat_text] * n_sample, return_tensors="pt").to(model.device)

#     generate_ids = model.generate(
#         inputs=model_inputs.input_ids,
#         attention_mask=model_inputs.attention_mask,
#         max_new_tokens=1024
#     )

#     input_length = model_inputs.input_ids.shape[1]
#     completions = []
    
#     for i in range(len(generate_ids)):
#         new_tokens = generate_ids[i][input_length:]
#         decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
#         final_code = decoded
#         if "```python" in decoded:
#             try:
#                 final_code = decoded.split("```python")[1].split("```")[0]
#             except:
#                 pass
#         elif "```" in decoded:
#             try:
#                 final_code = decoded.split("```")[1].split("```")[0]
#             except:
#                 pass
            
#         final_code = final_code.rstrip().lstrip('\n')
        
#         if "def " in final_code:
#             full_code = final_code
#         else:
#             full_code = prompt + "\n" + final_code
            
#         completions.append(full_code)
#     return completions

# def main():
#     collab_model, collab_tokenizer = load_collaborative_model()

#     output_dir = "./results"
#     os.makedirs(output_dir, exist_ok=True)
#     save_path = os.path.join(output_dir, "collab_result.json")
    
#     prompts_file = "path/to/prompts.jsonl"
#     print(f"\n[2/3] Reading prompt file {prompts_file}...")
#     with open(prompts_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     enamel_results = {}
#     print("\n[3/3] Starting collaborative generation...")
    
#     first_print_done = False
    
#     for line in tqdm(lines, desc="Collaborative Generation"):
#         data = json.loads(line.strip())
#         task_id_str = data["task_id"]
#         prompt_text = data["prompt"]
        
#         problem_idx = task_id_str.split("/")[-1]
        
#         completions = generate_collaborative_completions(collab_model, collab_tokenizer, prompt_text, NUM_SAMPLES)
#         enamel_results[str(problem_idx)] = completions
        
#         if not first_print_done:
#             print(f"\n[Preview] First task ({task_id_str}) generated:\n{enamel_results[str(problem_idx)][0][:300]}...\n")
#             first_print_done = True

#     with open(save_path, 'w', encoding='utf-8') as f:
#         json.dump(enamel_results, f, ensure_ascii=False, indent=2)
        
#     print("\n" + "="*60)
#     print(f"Done. Results saved to: {save_path}")
#     print("="*60 + "\n")

# if __name__ == "__main__":
#     main()