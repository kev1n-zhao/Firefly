from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = 'Qwen/Qwen-7B'
    adapter_name_or_path = 'output/firefly-qwen-7b'
    save_path = 'output/firefly-qwen-7b-qlora-merge'

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        device_map='auto',
        max_memory=16,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
