import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sberbank-ai/rugpt3large_based_on_gpt2"
MODEL_PATH = Path("rugpt")

# TARGET_PAIR = (..., ...)
PROMPT = "Томский государственный университет является одним из самых "


def load_model():
    if not MODEL_PATH.exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    return model, tokenizer


def generate(
    model,
    tok,
    text,
    do_sample=True,
    max_length=100,
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.95,
    temperature=1,
    num_beams=1,
    no_repeat_ngram_size=5,
):
    input_ids = tok.encode(text, return_tensors="pt").to(model.device)
    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    decoded_outputs = []
    for sample in out:
        decoded = tok.decode(sample, skip_special_tokens=True)
        decoded_outputs.append(decoded)

    return decoded_outputs


if __name__ == "__main__":
    torch.manual_seed(42)

    model, tokenizer = load_model()

    outputs = generate(
        model,
        tokenizer,
        PROMPT,
        do_sample=True,
        max_length=512,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=5.0,
    )

    generated = outputs[0]
    print(PROMPT)
    print()
    print(generated)
