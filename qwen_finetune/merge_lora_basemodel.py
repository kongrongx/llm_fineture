from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModel

# 加载已经微调过的LoRA模型
base_model_path = 'G:/WorkSpace/aigc/llm_models_store/llm_chat_models/qwen2_0.5b_instruct'
lora_model_path = 'qwen_output/checkpoint-5000'

base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModel.from_pretrained(
    base_model_path,
    trust_remote_code=True,
)


model = PeftModel.from_pretrained(base_model, lora_model_path)



# 合并LoRA权重到基础模型中
merged_model = model.merge_and_unload()

# 保存合并后的模型
output_path = 'G:/WorkSpace/aigc/llm_models_store/llm_chat_models/qwen2_0.5b_instruct_self_condition'
merged_model.save_pretrained(output_path)


base_model_tokenizer.save_pretrained(output_path)


