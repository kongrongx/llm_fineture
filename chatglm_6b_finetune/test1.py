from transformers import AutoTokenizer, AutoModel

# cache_dir = 'G:/WorkSpace/aigc/llm_models_store/llm_chat_models/chatglm3_6b'
cache_dir = '/home/wuyou/llm_finetune/llm_models_store/ZhipuAI/chatglm3-6b'

tokenizer = AutoTokenizer.from_pretrained(cache_dir, trust_remote_code=True)
# model = AutoModel.from_pretrained(cache_dir, trust_remote_code=True, device='cpu')
model = AutoModel.from_pretrained(
    cache_dir,
    # load_in_8bit=True,
    # load_in_4bit=True,
    trust_remote_code=True,
    # device='cpu'
    device='cuda'
)
model = model.eval()


response, history = model.chat(tokenizer, "你是谁", history=[])
print(response)
