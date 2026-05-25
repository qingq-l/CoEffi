import torch
import copy
from typing import Optional, Union
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    PreTrainedModel,
    LlamaForCausalLM,
    CodeGenForCausalLM,
    GPTBigCodeForCausalLM,
    Qwen2ForCausalLM,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor  
)
from transformers.generation.utils import GenerateOutput
import torch.nn.functional as F

class ModelWithExperts(PreTrainedModel):

    def _build_custom_processor(self, generation_config, input_ids):
        processors = LogitsProcessorList()
        rep_penalty = getattr(generation_config, 'repetition_penalty', 1.0)
        if rep_penalty is not None and rep_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=rep_penalty))
        return processors

    def _build_custom_warper(self, generation_config):
        warpers = LogitsProcessorList()
        temp = getattr(generation_config, 'temperature', 1.0)
        if temp is not None and temp != 1.0:
            warpers.append(TemperatureLogitsWarper(temp))
        top_k = getattr(generation_config, 'top_k', 0)
        if top_k is not None and top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
        top_p = getattr(generation_config, 'top_p', 1.0)
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
        return warpers

    @torch.no_grad()
    def generate_with_experts(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        expert_model = kwargs.pop('expert_lm', None)
        expert_kwargs = kwargs.pop('model_kwargs_expert', {})
        
        initial_threshold = kwargs.pop('threshold', 0.5)
        window_size = kwargs.pop('window_size', 10)
        min_threshold = kwargs.pop('min_threshold', 0.3)
        max_threshold = kwargs.pop('max_threshold', 1.2)
        adj_step = kwargs.pop('adj_step', 0.1)

        expert_temp = kwargs.pop('expert_temperature', 1.0)
        expert_top_p = kwargs.pop('expert_top_p', 1.0)
        max_new_tokens = kwargs.pop('max_new_tokens', 100)
        pad_token_id_arg = kwargs.pop('pad_token_id', None)
        eos_token_id_arg = kwargs.pop('eos_token_id', None)
        repetition_penalty = kwargs.pop('repetition_penalty', 1.0)

        if generation_config is None:
            generation_config = self.generation_config
        if generation_config is None:
            generation_config = GenerationConfig()
        generation_config = copy.deepcopy(generation_config)
        
        generation_config.update(**kwargs)
        generation_config.repetition_penalty = repetition_penalty

        pad_token_id = pad_token_id_arg if pad_token_id_arg is not None else generation_config.pad_token_id
        eos_token_id = eos_token_id_arg if eos_token_id_arg is not None else generation_config.eos_token_id

        model_kwargs = kwargs

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        custom_processors = self._build_custom_processor(generation_config, inputs)
        logits_processor.extend(custom_processors)

        logits_warper = self._build_custom_warper(generation_config)
        expert_generation_config = copy.deepcopy(generation_config)
        expert_generation_config.temperature = expert_temp
        expert_generation_config.top_p = expert_top_p
        logits_warper_expert = self._build_custom_warper(expert_generation_config)

        return self.sample_with_experts(
            inputs,
            expert_model=expert_model,
            expert_kwargs=expert_kwargs,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            logits_warper_expert=logits_warper_expert,
            initial_threshold=initial_threshold,
            window_size=window_size,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            adj_step=adj_step,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            **model_kwargs
        )

    def sample_with_experts(
        self,
        input_ids,
        expert_model,
        expert_kwargs,
        logits_processor,
        logits_warper,
        logits_warper_expert,
        initial_threshold,
        window_size,
        min_threshold,
        max_threshold,
        adj_step,
        stopping_criteria,
        pad_token_id,
        eos_token_id,
        max_new_tokens,
        **model_kwargs,
    ):
        if expert_model is None:
            raise ValueError("❌ expert_model is None!")

        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if "cache_position" not in model_kwargs:
            model_kwargs["cache_position"] = torch.arange(seq_length, device=device)
        if "cache_position" not in expert_kwargs:
            expert_kwargs["cache_position"] = torch.arange(seq_length, device=device)
            
        model_kwargs["use_cache"] = True
        expert_kwargs["use_cache"] = True

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        expert_model_kwargs = expert_kwargs.copy()

        dynamic_threshold = torch.full((batch_size, 1), initial_threshold, device=device, dtype=torch.float)
        
        history_agreement = torch.ones((batch_size, window_size), device=device, dtype=torch.float)
        step_idx = 0

        for _ in range(max_new_tokens):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            expert_inputs = expert_model.prepare_inputs_for_generation(input_ids, **expert_model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            outputs_expert = expert_model(**expert_inputs, return_dict=True)
            next_token_logits_expert = outputs_expert.logits[:, -1, :]

            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            
            expert_scores = logits_warper_expert(input_ids, next_token_logits_expert)

            probs_main = F.softmax(next_token_scores, dim=-1)
            probs_expert = F.softmax(expert_scores, dim=-1)

            expert_token_proposal = torch.multinomial(probs_expert, num_samples=1)
            p_main_at_token = torch.gather(probs_main, 1, expert_token_proposal)
            p_expert_at_token = torch.gather(probs_expert, 1, expert_token_proposal)

            ratio = p_main_at_token / (p_expert_at_token + 1e-8) 
            
            current_agreement = torch.clamp(ratio, max=1.0)
            
            history_agreement[:, step_idx % window_size] = current_agreement.squeeze(-1)
            step_idx += 1
            
            avg_agreement = history_agreement.mean(dim=-1, keepdim=True)

            increase_mask = avg_agreement < 0.3
            decrease_mask = avg_agreement > 0.8

            dynamic_threshold = torch.where(increase_mask, dynamic_threshold + adj_step, dynamic_threshold)
            dynamic_threshold = torch.where(decrease_mask, dynamic_threshold - adj_step, dynamic_threshold)
            
            dynamic_threshold = torch.clamp(dynamic_threshold, min=min_threshold, max=max_threshold)

            u = torch.rand(batch_size, 1, device=device)
            u_adjusted = u * dynamic_threshold 
            
            accept_mask = u_adjusted < torch.min(torch.ones_like(ratio), ratio)
            token_from_main = torch.multinomial(probs_main, num_samples=1)
            final_token = torch.where(accept_mask, expert_token_proposal, token_from_main)
            final_token = final_token.squeeze(1)

            if eos_token_id is not None:
                next_tokens = final_token * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            else:
                next_tokens = final_token

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            expert_model_kwargs = expert_model._update_model_kwargs_for_generation(
                outputs_expert, expert_model_kwargs, is_encoder_decoder=expert_model.config.is_encoder_decoder
            )

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).long())
                if unfinished_sequences.max() == 0:
                    break

        return input_ids


class CodeLlamaModelLM(LlamaForCausalLM, ModelWithExperts):
    def __init__(self, config):
        super().__init__(config)
class CodegenModelLM(CodeGenForCausalLM, ModelWithExperts):
    def __init__(self, config):
        super().__init__(config)
class StarcodeModelLM(GPTBigCodeForCausalLM, ModelWithExperts):
    def __init__(self, config):
        super().__init__(config)
class Qwen2ModelLM(Qwen2ForCausalLM, ModelWithExperts):
    def __init__(self, config):
        super().__init__(config)
class DeepSeekModelLM(LlamaForCausalLM, ModelWithExperts):
    def __init__(self, config):
        super().__init__(config)