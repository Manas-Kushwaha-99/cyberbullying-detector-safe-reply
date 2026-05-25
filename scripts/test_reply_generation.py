"""Test fine-tuned FLAN-T5-small reply generator."""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.config import DEVICE, MODELS_DIR

print("Loading fine-tuned FLAN-T5-small reply generator...")
model_path = os.path.join(MODELS_DIR, "flan_t5_small_reply")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
model.eval()

prefix = "generate safe reply: "

def generate_reply(text, num_beams=4, max_length=128):
    input_text = prefix + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Predefined test cases
TEST_CASES = [
    "Go back to where you came from, nobody wants you here.",
    "Women are way too emotional to ever hold a leadership position.",
    "Your religion is a violent cult that needs to be banned.",
    "Girls can't understand technology properly.",
    "People from your race are always ruining this country.",
    "You'll never fit in because of who you are.",
    "Your community is always causing trouble.",
    "Stop pretending to be something you're not.",
    "Your beliefs are completely stupid.",
    "Nobody takes women gamers seriously.",
]

print("\n" + "=" * 70)
print("REPLY GENERATION TESTS")
print("=" * 70)

for text in TEST_CASES:
    reply = generate_reply(text)
    print(f"\nCyberbullying: {text}")
    print(f"Safe Reply:    {reply}")

print("\n" + "=" * 70)
print("Interactive Mode - Type a cyberbullying message to get a reply")
print("Type 'quit' to exit")
print("=" * 70)

while True:
    try:
        text = input("\n>>> ")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break
    if not text or text.lower() in ("quit", "exit", "q"):
        print("Exiting...")
        break
    reply = generate_reply(text)
    print(f"Reply: {reply}")
