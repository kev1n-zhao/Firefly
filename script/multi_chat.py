from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = 'Qwen/Qwen-7B'
    adapter_name_or_path = 'output/firefly-qwen-7b/final'
    save_path = 'output/firefly-qwen-7b/merged'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path,
    #     trust_remote_code=True,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     device_map='auto'
    # )
    max_memory = {i: '12000MB' for i in range(1)}
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        offload_folder="output/firefly-qwen-7b/offload",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_8bit=True

        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        # ),
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    device = 'cuda'
    max_new_tokens = 500    # 每轮对话最多生成多少个token
    history_max_len = 1000  # 模型记忆的最大token长度
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0


    # 开始对话
    utterance_id = 0    # 记录当前是第几轮对话，为了契合chatglm的数据组织格式
    user_input = input('User：')
    while True:
        utterance_id += 1
        # chatglm使用官方的数据组织格式
        if model.config.model_type == 'chatglm':
            user_input = '[Round {}]\n\n问：{}\n\n答：'.format(utterance_id, user_input)
            user_input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
        # firefly的数据组织格式
        # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
        else:
            input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
            eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
            user_input_ids = torch.concat([input_ids, eos_token_id], dim=1)
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        model_input_ids = history_token_ids[:, -history_max_len:].to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
            )
        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
        response = tokenizer.batch_decode(response_ids)
        print("Firefly：" + response[0].strip().replace(tokenizer.eos_token, ""))
        user_input = input('User：')


if __name__ == '__main__':
    merge_lora_to_base_model()
