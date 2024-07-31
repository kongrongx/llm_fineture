from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModel


def transformer_download(cache_dir='llm_models_store'):
    # 指定模型名和下载目录
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir=cache_dir)
    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)


def modelscope_download(cache_dir='llm_models_store'):
    from modelscope import snapshot_download
    model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0", cache_dir=cache_dir)


if __name__ == '__main__':
    modelscope_download()