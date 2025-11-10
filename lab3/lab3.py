import os
from transformers import BertTokenizer, BertForMaskedLM, pipeline

MODEL = "DeepPavlov/rubert-base-cased"
MODEL_PATH = "./bert"

if not os.path.exists(MODEL_PATH):
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    model = BertForMaskedLM.from_pretrained(MODEL)

    tokenizer.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForMaskedLM.from_pretrained(MODEL_PATH)

    # Взял первую строку из википедии про ТГУ (с некоторыми изменениями) и попробовал сделать так, чтобы на месте [MASK] модель предсказывала древний/старый
    text = "Томский государственный университет - это самый [MASK] университет, построенный на территории Западной Сибири в 1878 году."

    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    top_words = unmasker(text, top_k=10)

    print(f"Исходное предложение: {text}")

    print("Топ10 наиболее вероятных слов на месте [MASK]:")
    for word_info in top_words:
        print(f"\t{word_info['token_str']}, prob: {word_info['score']:.6f}")
