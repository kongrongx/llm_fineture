from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Annotated, Union

import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer


def merge_lora_baasemodel_v1(base_model_path,lora_model_path,merge_model_path):
    base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    # 合并LoRA权重到基础模型中
    merged_model = model.merge_and_unload()
    # 保存合并后的模型
    merged_model.save_pretrained(merge_model_path)
    base_model_tokenizer.save_pretrained(merge_model_path)


def merge_lora_baasemodel_v2(lora_model_path, merge_model_path):
    #lora_model里的adapter_config.json自带了原始模型的位置，所以不需要在此输入原始模型
    merged_model, tokenizer = load_model_and_tokenizer(lora_model_path)
    # 保存合并后的模型
    merged_model.save_pretrained(merge_model_path)
    tokenizer.save_pretrained(merge_model_path)



def main():
    # 加载已经微调过的LoRA模型
    base_model_path = 'G:/WorkSpace/aigc/llm_models_store/llm_chat_models/chatglm3_6b'
    lora_model_path = 'output/checkpoint-3000'
    merge_model_path = 'G:/WorkSpace/aigc/llm_models_store/llm_chat_models/chatglm3_6b_lore_self_condition'

    merge_lora_baasemodel_v1(base_model_path,lora_model_path,merge_model_path)
    # merge_lora_baasemodel_v2()

if __name__ == '__main__':
    main()