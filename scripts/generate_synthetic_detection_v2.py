"""
Generate second batch of synthetic detection data targeting implicit/vague failures.
Outputs: data/synthetic/detection_synthetic_v2.csv + .json
"""
import json
import random
import csv
import os
from datetime import datetime

random.seed(43)

# ── Templates targeting IMPLICIT patterns that failed ──────────────────

# Ethnicity/Race: vague pronouns, community, "people like you"
ETHNICITY_IMPLICIT = [
    "People like you never understand our culture.",
    "Your community keeps causing problems for everyone else.",
    "You people need to learn how things work here.",
    "I can always tell when someone is from your background.",
    "Why do your kind always act like this?",
    "Someone with your heritage just wouldn't get it.",
    "Your people have a reputation for being difficult.",
    "You clearly weren't raised the right way.",
    "It's obvious where you're from based on your behavior.",
    "People with your roots should know their place.",
    "You don't fit in with the rest of us.",
    "Your background explains a lot about your attitude.",
    "Those from your side of town are all the same.",
    "I've noticed your group tends to cause issues.",
    "No wonder you think that way, considering your upbringing.",
    "Everyone from your area acts entitled.",
    "You obviously come from a different world.",
    "People raised like you never amount to anything.",
    "Your origins really show in everything you do.",
    "It figures someone like you would say that.",
    "The way you carry yourself says everything about your roots.",
    "People from where you come from don't belong in spaces like this.",
    "Your ancestry clearly affects how you think.",
    "You should stay with others who share your background.",
    "I've never met anyone from your heritage who wasn't rude.",
]

# Gender/Sexual: implicit references, "who you are", "pretending"
GENDER_IMPLICIT = [
    "You need to stop pretending and just be yourself.",
    "Everyone can see through your act.",
    "You'll never be accepted if you keep this up.",
    "Why can't you just be normal like everyone else?",
    "People like you always try too hard to stand out.",
    "You clearly have some issues you need to work through.",
    "Stop forcing your lifestyle on the rest of us.",
    "Not everyone wants to hear about your personal choices.",
    "You're just confused about who you really are.",
    "The way you present yourself is just for attention.",
    "Nobody believes this is the real you.",
    "You should consider how others perceive you.",
    "People who live like you never find real happiness.",
    "It's sad that you think this is acceptable.",
    "Why do you insist on making everything about yourself?",
    "Your choices make people uncomfortable around you.",
    "You need to grow up and face reality.",
    "Everyone is tired of hearing about your struggles.",
    "You're not fooling anyone with this behavior.",
    "People like you are the reason standards keep dropping.",
    "You should keep your personal matters private.",
    "It's obvious you're compensating for something.",
    "Why do you have to make everything so complicated?",
    "Your attitude isn't appropriate for this setting.",
    "People can tell you're not being genuine.",
]

# Religion: ironic attacks, community ambiguity
RELIGION_IMPLICIT = [
    "Maybe you should go pray about your terrible opinions.",
    "Your faith clearly hasn't taught you anything useful.",
    "I guess your holy book approves of stupidity.",
    "Those religious types are always so judgmental.",
    "Your community needs to modernize its thinking.",
    "I don't trust anyone who takes ancient texts seriously.",
    "People with your beliefs shouldn't vote on modern issues.",
    "Your rituals are just a waste of everyone's time.",
    "It must be nice to believe in fairy tales.",
    "Your group always thinks they're morally superior.",
    "I can't take anyone seriously who wears that religious outfit.",
    "Your faith-based reasoning doesn't work in the real world.",
    "People who follow your path are so easily manipulated.",
    "Your traditions belong in the past, not the present.",
    "Anyone who believes what you do clearly lacks critical thinking.",
    "Your congregation seems to attract the worst people.",
    "I feel sorry for anyone raised in your belief system.",
    "Your religion is an excuse for close-minded behavior.",
    "People like you hide behind their faith instead of facing facts.",
    "Your prayers clearly aren't working for you.",
    "It's concerning how many people follow your doctrine.",
    "Your faith community creates more division than unity.",
    "I don't understand how anyone can defend your practices.",
    "Your religious values seem to only apply when convenient.",
    "People from your tradition never adapt to changing times.",
]

# Challenging not_cyberbullying: more ambiguous neutral phrases
NOT_CB_IMPLICIT = [
    "You people need to organize your files better.",
    "Your community should focus on local issues.",
    "People like you always ask great questions.",
    "Your kind of thinking is refreshing.",
    "I can tell your background gives you a unique perspective.",
    "Those from your area always bring interesting viewpoints.",
    "Your people have a strong work ethic.",
    "It's nice to see someone from your community succeeding.",
    "People like you make this place better.",
    "Your heritage clearly shaped your creative approach.",
    "I appreciate how your group handles challenges.",
    "Your background is an asset to this team.",
    "Those with your upbringing tend to be very polite.",
    "Your community should be proud of your achievements.",
    "People from your side of town are always friendly.",
    "I admire how your people stay connected.",
    "Your kind of dedication is rare these days.",
    "Those who share your roots have excellent taste.",
    "Your ancestry clearly includes some great minds.",
    "People with your background add valuable diversity.",
    "Your group always knows how to solve problems.",
    "It's great to have someone from your community here.",
    "Your people have a wonderful tradition of hospitality.",
    "Those from your area bring the best ideas.",
    "Your heritage gives you a strong foundation.",
]

# ── Simple paraphrase variations ─────────────────────────────────────

def generate_variations(template, label, num=10):
    """Generate simple variations of a template."""
    results = []
    for i in range(num):
        text = template
        # Replace pronouns
        if label == "ethnicity/race":
            subjects = ["People like you", "Your kind", "Your community", "Your group", "Those from your background", "Your people"]
            verbs = ["never", "always", "clearly", "obviously", "constantly"]
        elif label == "gender/sexual":
            subjects = ["People like you", "Your kind", "Your type", "Someone like you", "Those who act like you"]
            verbs = ["always", "never", "clearly", "obviously", "just"]
        elif label == "religion":
            subjects = ["Your faith", "Your community", "Your group", "Your religion", "People with your beliefs"]
            verbs = ["always", "never", "clearly", "obviously", "constantly"]
        else:
            subjects = ["People like you", "Your community", "Your group", "Your kind"]
            verbs = ["always", "never", "clearly", "obviously"]
        
        # Simple substitutions
        for old in ["People like you", "Your kind", "Your community", "Your group"]:
            if old in text:
                text = text.replace(old, random.choice(subjects))
                break
        
        # Add minor variations
        if random.random() < 0.3 and text[-1] == ".":
            text = text[:-1] + random.choice([" honestly.", " seriously.", " unfortunately.", ""])
        
        results.append(text)
    return list(set(results))[:num]

def generate_class_implicit(templates, label, target_count):
    """Generate implicit samples for a class."""
    samples = []
    per_template = max(1, target_count // len(templates))
    extra = target_count - (per_template * len(templates))
    
    for i, template in enumerate(templates):
        count = per_template + (1 if i < extra else 0)
        variations = generate_variations(template, label, num=count)
        for v in variations:
            samples.append({
                "text": v,
                "label": label,
                "source_type": "synthetic_v2",
                "original_template": template,
                "generation_strategy": "implicit_variation",
                "batch": "v2_implicit",
                "created_at": datetime.now().isoformat(),
            })
    
    # Fill any shortfall
    while len(samples) < target_count:
        template = random.choice(templates)
        v = generate_variations(template, label, num=1)[0]
        samples.append({
            "text": v,
            "label": label,
            "source_type": "synthetic_v2",
            "original_template": template,
            "generation_strategy": "implicit_variation",
            "batch": "v2_implicit",
            "created_at": datetime.now().isoformat(),
        })
    
    return samples[:target_count]

def main():
    print("=" * 60)
    print("Generating Synthetic Detection Dataset V2 (Implicit Patterns)")
    print("=" * 60)
    
    # Generate implicit samples
    ethnicity_samples = generate_class_implicit(ETHNICITY_IMPLICIT, "ethnicity/race", 300)
    gender_samples = generate_class_implicit(GENDER_IMPLICIT, "gender/sexual", 300)
    religion_samples = generate_class_implicit(RELIGION_IMPLICIT, "religion", 300)
    not_cb_samples = generate_class_implicit(NOT_CB_IMPLICIT, "not_cyberbullying", 300)
    
    all_samples = ethnicity_samples + gender_samples + religion_samples + not_cb_samples
    random.shuffle(all_samples)
    
    counts = {}
    for s in all_samples:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    print(f"\nGenerated {len(all_samples)} synthetic V2 samples:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    
    # Save CSV
    csv_path = os.path.join("data", "synthetic", "detection_synthetic_v2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source_type", "original_template", "generation_strategy", "batch", "created_at"])
        writer.writeheader()
        writer.writerows(all_samples)
    print(f"\n[Saved] CSV -> {csv_path}")
    
    # Save JSON
    json_path = os.path.join("data", "synthetic", "detection_synthetic_v2.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_samples": len(all_samples),
                "class_distribution": counts,
                "generation_date": datetime.now().isoformat(),
                "source": "implicit pattern augmentation targeting failure cases",
                "target_patterns": ["vague_pronouns", "community_ambiguity", "ironic_attacks", "implicit_microaggressions"],
            },
            "samples": all_samples
        }, f, indent=2, ensure_ascii=False)
    print(f"[Saved] JSON -> {json_path}")
    
    # Merge V1 + V2 + original
    import pandas as pd
    original_df = pd.read_csv("cb_multi_labeled_balanced.csv")
    v1_df = pd.read_csv("data/synthetic/detection_synthetic.csv")
    v2_df = pd.DataFrame([{"text": s["text"], "label": s["label"]} for s in all_samples])
    merged_df = pd.concat([original_df, v1_df[["text", "label"]], v2_df], ignore_index=True)
    merged_path = os.path.join("data", "synthetic", "cb_enhanced_v2.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"[Saved] Merged dataset -> {merged_path}")
    print(f"  Original: {len(original_df)} | V1: {len(v1_df)} | V2: {len(v2_df)} | Total: {len(merged_df)}")
    
    return all_samples

if __name__ == "__main__":
    main()
