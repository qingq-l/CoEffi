# coding: utf-8
#This part is the configuration adapted for deepseek-coder, qwen2.5-coder, and opencoder.
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

class EnamelGenerator(object):
    def __init__(self, model_name_or_path, save_name=None) -> None:
        self.model_name_or_path = model_name_or_path
        self.save_name = save_name if save_name else "baseline_result"
        
        self.output_dir = "./results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Loading Model [{self.save_name}] onto Multi-GPUs from local path...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

    def generate_completions(self, prompt: str, n_sample=1):
        messages = [
            {"role": "system", "content": "You are an world-class Python programmer. Please provide the complete, runnable Python code to solve the problem."},
            {"role": "user", "content": f"Please provide the complete Python solution. Include the imports, function signature, and full implementation. Output your code inside a ```python block.\n\n```python\n{prompt}\n```"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text] * n_sample, return_tensors="pt").to("cuda")

        generate_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
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
                try:
                    final_code = decoded.split("```python")[1].split("```")[0]
                except: pass
            elif "```" in decoded:
                try:
                    final_code = decoded.split("```")[1].split("```")[0]
                except: pass
                
            final_code = final_code.rstrip().lstrip('\n')
            
            if "def " in final_code:
                full_code = final_code
            else:
                full_code = prompt + "\n" + final_code
                
            completions.append(full_code)
            
        return completions

    def generate_all(self, prompts_file="enamel_prompts.jsonl", num_samples=1):
        enamel_results = {}
        
        print(f"Reading prompts from {prompts_file}...")
        with open(prompts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Generating Codes"):
            data = json.loads(line.strip())
            task_id_str = data["task_id"]
            prompt_text = data["prompt"]
            
            problem_idx = task_id_str.split("/")[-1]
            completions = self.generate_completions(prompt_text, num_samples)
            enamel_results[str(problem_idx)] = completions

        save_path = os.path.join(self.output_dir, f'{self.save_name}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(enamel_results, f, ensure_ascii=False, indent=2)
            
        print(f"Done! Results saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple ENAMEL Generation')
    parser.add_argument('--model_name_or_path', default="path/to/model", help="Absolute path to your local model")
    parser.add_argument('--save_name', type=str, default="baseline_result", help="Custom name for saving output files")
    parser.add_argument('--samples', type=int, default=1, help="generation samples per task")
    parser.add_argument('--prompts_file', type=str, default="path/to/prompts.jsonl", help="The jsonl file containing the prompts")
    
    args = parser.parse_args()
        
    print(f"Current model: [{args.model_name_or_path}]")
    
    generator = EnamelGenerator(args.model_name_or_path, args.save_name)
    generator.generate_all(prompts_file=args.prompts_file, num_samples=args.samples)



# # coding: utf-8
# # The following part is the configuration adapted for StarCoder2.
# import os
# import json
# import torch
# import argparse
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM

# os.environ["TOKENIZERS_PARALLELISM"] = 'false'

# class EnamelGenerator(object):
#     def __init__(self, model_name_or_path, save_name=None) -> None:
#         self.model_name_or_path = model_name_or_path
#         self.save_name = save_name if save_name else "baseline_result"
        
#         self.output_dir = "./results"
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         print(f"Loading Model [{self.save_name}] onto Multi-GPUs from local path...")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.padding_side = 'left'
            
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name_or_path,
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             device_map="auto",
#         )

#     def generate_completions(self, prompt: str, n_sample=1):
#         enhanced_prompt = (
#             "You are a world-class Python programmer. Please provide the complete, runnable Python code to solve the problem. "
#             "Include the imports, function signature, and full implementation. Output your code inside a ```python block.\n\n"
#             f"```python\n{prompt}\n```"
#         )
        
#         text = (
#             "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n"
#             f"{enhanced_prompt}\n\n"
#             "### Response:\n"
#         )
        
#         model_inputs = self.tokenizer([text] * n_sample, return_tensors="pt").to(self.model.device)

#         generate_ids = self.model.generate(
#             **model_inputs,
#             max_new_tokens=1024,
#             do_sample=True,
#             top_p=0.95,
#             temperature=0.6,
#             eos_token_id=self.tokenizer.eos_token_id, 
#             pad_token_id=self.tokenizer.pad_token_id
#         )

#         input_length = model_inputs.input_ids.shape[1]
#         completions = []
        
#         for i in range(len(generate_ids)):
#             new_tokens = generate_ids[i][input_length:]
#             decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
#             final_code = decoded
#             if "```python" in decoded:
#                 try:
#                     final_code = decoded.split("```python")[1].split("```")[0]
#                 except: pass
#             elif "```" in decoded:
#                 try:
#                     final_code = decoded.split("```")[1].split("```")[0]
#                 except: pass
                
#             final_code = final_code.rstrip().lstrip('\n')
            
#             if "def " in final_code:
#                 full_code = final_code
#             else:
#                 full_code = prompt + "\n" + final_code
                
#             completions.append(full_code)
            
#         return completions

#     def generate_all(self, prompts_file="enamel_prompts.jsonl", num_samples=1):
#         enamel_results = {}
        
#         print(f"Reading prompts from {prompts_file}...")
#         with open(prompts_file, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
            
#         for line in tqdm(lines, desc="Generating Codes"):
#             data = json.loads(line.strip())
#             task_id_str = data["task_id"]
#             prompt_text = data["prompt"]
            
#             problem_idx = task_id_str.split("/")[-1]
#             completions = self.generate_completions(prompt_text, num_samples)
#             enamel_results[str(problem_idx)] = completions

#         save_path = os.path.join(self.output_dir, f'{self.save_name}.json')
#         with open(save_path, 'w', encoding='utf-8') as f:
#             json.dump(enamel_results, f, ensure_ascii=False, indent=2)
            
#         print(f"Done! Results saved to: {save_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Simple ENAMEL Generation')
#     parser.add_argument('--model_name_or_path', default="path/to/model", help="Absolute path to your local model")
#     parser.add_argument('--save_name', type=str, default="baseline_result", help="Custom name for saving output files")
#     parser.add_argument('--samples', type=int, default=1, help="generation samples per task")
#     parser.add_argument('--prompts_file', type=str, default="path/to/prompts.jsonl", help="The jsonl file containing the prompts")
    
#     args = parser.parse_args()
        
#     print(f"Current model: [{args.model_name_or_path}]")
    
#     generator = EnamelGenerator(args.model_name_or_path, args.save_name)
#     generator.generate_all(prompts_file=args.prompts_file, num_samples=args.samples)