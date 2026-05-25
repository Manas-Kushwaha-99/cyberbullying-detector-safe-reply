"""Extended test of fine-tuned FLAN-T5-small reply generator."""
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

INSTRUCTION = "respond to this hate speech with a calm, respectful de-escalation: "

def generate_reply(text, num_beams=4, max_length=128):
    input_text = INSTRUCTION + text
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

# Extended test cases covering all categories + edge cases
TEST_CASES = [
    # Ethnicity/Race
    ("Go back to where you came from, nobody wants you here.", "ethnicity/race"),
    ("People from your race are always ruining this country.", "ethnicity/race"),
    ("Your culture is completely inferior to ours.", "ethnicity/race"),
    ("You people are all the same.", "ethnicity/race"),
    ("Your accent makes you sound uneducated.", "ethnicity/race"),
    ("No wonder you failed, people like you never succeed.", "ethnicity/race"),
    ("Your community should stay off social media.", "ethnicity/race"),
    
    # Gender/Sexual
    ("Go back to the kitchen and leave the real work to the men.", "gender/sexual"),
    ("Women are way too emotional to ever hold a leadership position.", "gender/sexual"),
    ("People with your sexual orientation are completely unnatural and wrong.", "gender/sexual"),
    ("Girls can't handle technical work.", "gender/sexual"),
    ("Boys who cry are weak and pathetic.", "gender/sexual"),
    ("Nobody takes women gamers seriously.", "gender/sexual"),
    ("Your gender identity is just for attention.", "gender/sexual"),
    ("Stop pretending to be something you're not.", "gender/sexual"),
    
    # Religion
    ("Your religion is a violent cult that needs to be banned.", "religion"),
    ("All people who practice your faith are completely brainwashed.", "religion"),
    ("Your religious beliefs are completely backwards and stupid.", "religion"),
    ("Your holy book is full of lies and promotes hatred.", "religion"),
    ("Go pray instead of embarrassing yourself online.", "religion"),
    ("Your religion should be banned everywhere.", "religion"),
    ("Only fools follow your religious practices.", "religion"),
    
    # Edge cases - implicit/microaggressions
    ("You'll never fit in because of who you are.", "gender/sexual"),
    ("Your community is always causing trouble.", "religion"),
    ("Only idiots would support someone like you.", "gender/sexual"),
    ("I can't believe anyone still follows that religion.", "religion"),
    ("Typical behavior from someone of your race.", "ethnicity/race"),
]

print("\n" + "=" * 80)
print("EXTENDED REPLY GENERATION TESTS")
print("=" * 80)

current_category = None
for text, category in TEST_CASES:
    if category != current_category:
        current_category = category
        print(f"\n{'='*80}")
        print(f"CATEGORY: {current_category.upper()}")
        print("=" * 80)
    
    reply = generate_reply(text)
    print(f"\nAggressor: {text}")
    print(f"Reply:     {reply}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
