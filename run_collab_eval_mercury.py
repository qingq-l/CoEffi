#This part is the configuration adapted for deepseek-coder, qwen2.5-coder, and opencoder.
import os
import sys
import json
import torch
import gc
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-mercury-check"

MERCURY_PATH = "path/to/mercury"
SKELDPO_PATH = "path/to/skeldpo"
sys.path.append(MERCURY_PATH)
sys.path.append(SKELDPO_PATH)
from src import evaluator as Evaluator
from CustomizedGeneration import ModelWithExperts

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

def custom_generate_completions(self, prompt: str, n_sample=1):
    messages = [
        {"role": "system", "content": "You are a world-class Python programmer. Please output the code solution directly within a code block."},
        {"role": "user", "content": prompt}
    ]
    
    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = self.tokenizer([text] * n_sample, return_tensors="pt").to("cuda")

    generate_ids = self.model.generate(
        inputs=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        temperature=MAIN_TEMP,
        eos_token_id=self.tokenizer.eos_token_id, 
        pad_token_id=self.tokenizer.eos_token_id
    )

    input_length = model_inputs.input_ids.shape[1]
    completions = []
    
    for i in range(len(generate_ids)):
        new_tokens = generate_ids[i][input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        final_code = decoded
        if "```python" in decoded:
            try: final_code = decoded.split("```python")[1].split("```")[0].strip()
            except: pass
        elif "```" in decoded:
            try: final_code = decoded.split("```")[1].split("```")[0].strip()
            except: pass
        
        if not hasattr(self, '_first_print_done'):
            print(f"\n[Preview] First generated code:\n{final_code[:200]}...\n")
            self._first_print_done = True
            
        completions.append(final_code)
    return completions

def custom_prompt_generate(self, instance):
    content = instance['pretty_content'][0]
    code_prompt = instance['prompt']
    return f"Please solve this problem:\n{content}\n\nCode context:\n{code_prompt}"

Evaluator.DistributeWiseEvaluator.generate_completions = custom_generate_completions
Evaluator.DistributeWiseEvaluator.prompt_generate = custom_prompt_generate

def load_collaborative_model():
    print(f"\n[1/3] Loading collaborative model...")
    
    main_model = AutoModelForCausalLM.from_pretrained(
        MAIN_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0", 
        trust_remote_code=True
    )
    
    main_model.__class__ = type(
        f"Collab_{main_model.__class__.__name__}", 
        (main_model.__class__, ModelWithExperts), 
        {}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    expert_base = AutoModelForCausalLM.from_pretrained(
        EXPERT_BASE_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    expert_model = PeftModel.from_pretrained(expert_base, EXPERT_ADAPTER_PATH)
    expert_model.eval()

    extra_eos = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    final_eos = extra_eos if extra_eos is not None else tokenizer.eos_token_id

    main_model.generate = partial(
        main_model.generate_with_experts,
        expert_lm=expert_model,
        threshold=INITIAL_THRESHOLD,
        window_size=WINDOW_SIZE,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        adj_step=ADJ_STEP,
        expert_temperature=EXPERT_TEMP,
        temperature=MAIN_TEMP,
        do_sample=True,
        eos_token_id=final_eos,
        pad_token_id=tokenizer.pad_token_id
    )
    
    return main_model, tokenizer

def main():
    collab_model, collab_tokenizer = load_collaborative_model()

    print(f"\n[2/3] Initializing Mercury evaluator...")
    evaluator = Evaluator.DistributeWiseEvaluator(model_name_or_path=MAIN_MODEL_PATH, do_generate=True)

    print("\nCleaning up redundant memory...")
    if hasattr(evaluator, 'model'): del evaluator.model
    if hasattr(evaluator, 'tokenizer'): del evaluator.tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("[3/3] Injecting collaborative model...")
    evaluator.model = collab_model
    evaluator.tokenizer = collab_tokenizer

    evaluator.save_name = "collab_mercury_result"

    print("\nStep 1: Generating code samples...")
    evaluator.generate(num_samples_per_task=NUM_SAMPLES)
    
    print("\nStep 2: Evaluating in sandbox...")
    metrics = evaluator.evaluate(num_samples_per_task=NUM_SAMPLES)
    
    print("\n" + "="*60)
    print("Collaborative Evaluation Result")
    print("-" * 60)
    if metrics:
        pass_1 = metrics.get('Pass@1', metrics.get('pass@1', 0))
        beyond_1 = metrics.get('Beyond@1', metrics.get('beyond@1', 0))
        print(f" Pass@1:   {pass_1:.4f}")
        print(f" Beyond@1: {beyond_1:.4f}")
    else:
        print(" Evaluation returned no valid scores.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()


#The following part is the configuration adapted for StarCoder2.
# import os
# import sys
# import json
# import torch
# import gc
# from functools import partial
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-mercury-check"

# MERCURY_PATH = "path/to/mercury"
# SKELDPO_PATH = "path/to/skeldpo"
# sys.path.append(MERCURY_PATH)
# sys.path.append(SKELDPO_PATH)
# from src import evaluator as Evaluator
# from CustomizedGeneration import ModelWithExperts

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

# def custom_generate_completions(self, prompt: str, n_sample=1):
#     enhanced_prompt = (
#         "You are a world-class Python programmer. Please output the code solution directly within a Python code block.\n\n"
#         f"{prompt}"
#     )
    
#     text = (
#         "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n"
#         f"{enhanced_prompt}\n\n"
#         "### Response:\n"
#     )
    
#     model_inputs = self.tokenizer([text] * n_sample, return_tensors="pt").to("cuda")

#     generate_ids = self.model.generate(
#         inputs=model_inputs.input_ids,
#         attention_mask=model_inputs.attention_mask,
#         max_new_tokens=1024,
#         do_sample=True,
#         top_p=0.95,
#         temperature=MAIN_TEMP,
#         eos_token_id=self.tokenizer.eos_token_id, 
#         pad_token_id=self.tokenizer.eos_token_id
#     )

#     input_length = model_inputs.input_ids.shape[1]
#     completions = []
    
#     for i in range(len(generate_ids)):
#         new_tokens = generate_ids[i][input_length:]
#         decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
#         final_code = decoded
#         if "```python" in decoded:
#             try: final_code = decoded.split("```python")[1].split("```")[0].strip()
#             except: pass
#         elif "```" in decoded:
#             try: final_code = decoded.split("```")[1].split("```")[0].strip()
#             except: pass
        
#         if not hasattr(self, '_first_print_done'):
#             print(f"\n[Preview] First generated code:\n{final_code[:300]}...\n")
#             self._first_print_done = True
            
#         completions.append(final_code)
#     return completions

# def custom_prompt_generate(self, instance):
#     content = instance['pretty_content'][0]
#     code_prompt = instance['prompt']
#     return f"Please solve this problem:\n{content}\n\nCode context:\n{code_prompt}"

# Evaluator.DistributeWiseEvaluator.generate_completions = custom_generate_completions
# Evaluator.DistributeWiseEvaluator.prompt_generate = custom_prompt_generate

# def load_collaborative_model():
#     print(f"\n[1/3] Loading collaborative model...")
    
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

#     expert_base = AutoModelForCausalLM.from_pretrained(
#         EXPERT_BASE_PATH,
#         torch_dtype=torch.float16,
#         device_map="cuda:0",
#         trust_remote_code=True
#     )
#     expert_model = PeftModel.from_pretrained(expert_base, EXPERT_ADAPTER_PATH)
#     expert_model.eval()

#     main_model.generate = partial(
#         main_model.generate_with_experts,
#         expert_lm=expert_model,
#         threshold=INITIAL_THRESHOLD,
#         window_size=WINDOW_SIZE,
#         min_threshold=MIN_THRESHOLD,
#         max_threshold=MAX_THRESHOLD,
#         adj_step=ADJ_STEP,
#         expert_temperature=EXPERT_TEMP,
#         temperature=MAIN_TEMP,
#         do_sample=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id
#     )
    
#     return main_model, tokenizer

# def main():
#     collab_model, collab_tokenizer = load_collaborative_model()

#     print(f"\n[2/3] Initializing Mercury evaluator...")
#     evaluator = Evaluator.DistributeWiseEvaluator(model_name_or_path=MAIN_MODEL_PATH, do_generate=True)

#     print("\nCleaning up redundant memory...")
#     if hasattr(evaluator, 'model'): del evaluator.model
#     if hasattr(evaluator, 'tokenizer'): del evaluator.tokenizer
#     torch.cuda.empty_cache()
#     gc.collect()

#     print("[3/3] Injecting collaborative model...")
#     evaluator.model = collab_model
#     evaluator.tokenizer = collab_tokenizer

#     evaluator.save_name = "collab_mercury_result"

#     print("\nStep 1: Generating code samples...")
#     evaluator.generate(num_samples_per_task=NUM_SAMPLES)
    
#     print("\nStep 2: Evaluating in sandbox...")
#     metrics = evaluator.evaluate(num_samples_per_task=NUM_SAMPLES)
    
#     print("\n" + "="*60)
#     print("Collaborative Evaluation Result")
#     print("-" * 60)
#     if metrics:
#         pass_1 = metrics.get('Pass@1', metrics.get('pass@1', 0))
#         beyond_1 = metrics.get('Beyond@1', metrics.get('beyond@1', 0))
#         print(f" Pass@1:   {pass_1:.4f}")
#         print(f" Beyond@1: {beyond_1:.4f}")
#     else:
#         print(" Evaluation returned no valid scores.")
#     print("="*60 + "\n")

# if __name__ == "__main__":
#     main()