from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
    # cache_dir: str = None,
    token: str = None,
    cuda: bool = False,
):
    """
    Load a model from Hugging Face model hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, token=token  # , cache_dir=cache_dir, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=token  # , cache_dir=cache_dir, use_safetensors=True
    )

    if cuda:
        model.to("cuda")

    return model, tokenizer
