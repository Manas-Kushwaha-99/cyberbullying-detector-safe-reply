"""
Generate synthetic detection data to improve model robustness on failure cases.
Outputs: data/synthetic/detection_synthetic.csv + .json
"""
import json
import random
import csv
import os
from datetime import datetime

random.seed(42)

# ── Templates from user-provided failure examples ──────────────────
ETHNICITY_TEMPLATES = [
    "Go back to where you came from, nobody wants you here.",
    "People of your race are always ruining this country.",
    "You only got accepted because of your ethnic background, not your talent.",
    "Your people shouldn't be allowed to live in this neighborhood.",
    "I hate dealing with people from your ethnic group.",
    "Your culture is completely inferior to ours.",
    "Everyone from your background acts exactly the same lazy way.",
    "You don't belong in this country, go home.",
    "We don't want your kind working in our company.",
    "Your race is the reason why crime rates are so high here.",
    "Stop bringing your foreign customs into our community.",
    "You will never be a real citizen here because of your ethnicity.",
    "People from your ethnicity are always so backward.",
    "Go back to where your race belongs.",
    "Nobody likes your kind here.",
    "Your accent sounds weird because of your background.",
    "People of your race are bad at studies.",
    "Why are people from your community always like this?",
    "You don't fit in because of your ethnicity.",
    "Your race ruins everything.",
    "Everyone from your region behaves badly.",
    "I can tell your ethnicity just by how you act.",
    "Your people should stay separate from us.",
    "No wonder you failed, people from your race are dumb.",
]

GENDER_TEMPLATES = [
    "Go back to the kitchen and leave the real work to the men.",
    "Women are way too emotional to ever hold a leadership position.",
    "You only got this promotion because of gender diversity quotas.",
    "People with your sexual orientation are completely unnatural and wrong.",
    "This is a man's field, girls shouldn't even bother trying.",
    "Your lifestyle choice is disgusting and shouldn't be shown in public.",
    "Men who act like you are completely weak and pathetic.",
    "You sound like a typical emotional woman who can't handle logic.",
    "People like you shouldn't be allowed to marry or have families.",
    "Stop trying to push your gender agenda onto everyone else.",
    "You are a man, start acting like one instead of crying.",
    "Nobody respects a woman who tries to speak up in this forum.",
    "Girls can't handle technical work.",
    "Boys shouldn't cry, that's weak.",
    "You act too feminine.",
    "No one takes women seriously in leadership.",
    "You're weird because of your sexuality.",
    "Men are naturally better at this.",
    "People like you shouldn't be accepted.",
    "Your gender makes you less capable.",
    "Stop acting like that, it's embarrassing.",
    "You don't look masculine enough.",
    "Women should just stay quiet.",
    "Your identity is the reason nobody likes you.",
]

RELIGION_TEMPLATES = [
    "Your religion is a violent cult that needs to be banned.",
    "All people who practice your faith are completely brainwashed.",
    "Your religious beliefs are completely backwards and stupid.",
    "Nobody wants your fake religion practiced in our neighborhood.",
    "People of your faith are untrustworthy and hate freedom.",
    "Your holy book is full of lies and promotes hatred.",
    "We need to kick everyone of your religion out of this country.",
    "You pray to a fake god and your rituals are ridiculous.",
    "Your religious clothing makes everyone around you uncomfortable.",
    "I refuse to respect anyone who follows that backward religion.",
    "Your faith is the root cause of all the trouble in the world today.",
    "Stop forcing your religious rules on civilized people.",
    "Your religion teaches strange things.",
    "People from your religion are always causing problems.",
    "Nobody trusts followers of your faith.",
    "Your beliefs are outdated.",
    "You shouldn't celebrate your religious festivals here.",
    "People of your religion don't belong in this country.",
    "Your community is ruining society.",
    "I avoid people from your religion.",
    "Your faith makes you dangerous.",
    "Everyone from your religion thinks the same way.",
    "Your religious practices are stupid.",
    "No one wants your religion around.",
]

NOT_CB_TEMPLATES = [
    "I completely disagree with your political stance on this issue.",
    "The weather forecast says it might rain heavily this afternoon.",
    "Please stop messaging me because I am not interested in chatting.",
    "That was a terrible movie and the plot made absolutely no sense.",
    "Can someone help me fix this formatting error in my code?",
    "I am so frustrated with how slow the internet connection is today.",
    "The referee made an awful call that cost our team the game.",
    "I don't like your attitude during our group meetings lately.",
    "Please review the attached document and send over your feedback.",
    "I think your argument lacks evidence and historical accuracy.",
    "The restaurant service was incredibly slow and the food arrived cold.",
    "Let's cancel the meeting today since half the team is out sick.",
    "I disagree with your opinion.",
    "Please submit the assignment on time.",
    "That joke was not funny.",
    "Can you improve the presentation design?",
    "I think your answer is incorrect.",
    "Let's discuss this calmly.",
    "You should practice more for better results.",
    "I don't want to join the group today.",
    "Please avoid spamming the chat.",
    "Your internet connection seems unstable.",
    "Thank you for helping with the project.",
    "Good luck for your exam.",
]

# ── Augmentation strategies ────────────────────────────────────────

SYNONYMS = {
    "people": ["folks", "individuals", "persons", "group", "community"],
    "always": ["constantly", "forever", "repeatedly", "all the time", "nonstop"],
    "completely": ["totally", "absolutely", "entirely", "utterly", "fully"],
    "bad": ["terrible", "awful", "horrible", "dreadful", "atrocious"],
    "stupid": ["foolish", "ignorant", "senseless", "mindless", "unintelligent"],
    "hate": ["detest", "despise", "loathe", "abhor", "can't stand"],
    "disgusting": ["repulsive", "revolting", "sickening", "nauseating", "vile"],
    "wrong": ["incorrect", "immoral", "unacceptable", "improper", "inappropriate"],
    "weird": ["strange", "odd", "bizarre", "unnatural", "peculiar"],
    "weak": ["fragile", "feeble", "powerless", "vulnerable", "pathetic"],
    "lazy": ["idle", "sluggish", "unproductive", "inactive", "indolent"],
    "dumb": ["stupid", "ignorant", "dense", "slow", "brainless"],
    "violent": ["brutal", "savage", "aggressive", "cruel", "vicious"],
    "fake": ["false", "phony", "fraudulent", "bogus", "counterfeit"],
    "dangerous": ["harmful", "threatening", "risky", "unsafe", "perilous"],
    "ruining": ["destroying", "wrecking", "devastating", "spoiling", "undermining"],
    "inferior": ["lesser", "substandard", "second-rate", "mediocre", "deficient"],
    "backwards": ["outdated", "primitive", "archaic", "obsolete", "old-fashioned"],
    "ridiculous": ["absurd", "laughable", "preposterous", "ludicrous", "nonsensical"],
    "trouble": ["problems", "issues", "conflict", "unrest", "turmoil"],
    "bother": ["annoy", "irritate", "disturb", "pester", "harass"],
    "refuse": ["decline", "reject", "deny", "disallow", "forbid"],
    "kick out": ["remove", "expel", "deport", "banish", "evict"],
    "don't belong": ["aren't welcome", "are out of place", "are foreign", "are alien"],
    "not allowed": ["forbidden", "prohibited", "banned", "barred", "excluded"],
    "nobody wants": ["no one likes", "everyone hates", "people reject", "society excludes"],
    "go back": ["return", "leave", "get out", "move away", "depart"],
    "shut up": ["be quiet", "stay silent", "keep your mouth shut", "don't speak"],
    "can't handle": ["are incapable of", "are too weak for", "fail at", "struggle with"],
    "too emotional": ["overly sensitive", "irrational", "hysterical", "unstable"],
    "real work": ["serious work", "actual work", "important tasks", "hard work"],
    "man's field": ["male domain", "boys' club", "male profession", "men's territory"],
    "lifestyle choice": ["personal preference", "way of life", "behavior", "conduct"],
    "gender agenda": ["feminist propaganda", "LGBTQ agenda", "identity politics", "social engineering"],
    "brainwashed": ["indoctrinated", "programmed", "conditioned", "manipulated"],
    "holy book": ["scripture", "religious text", "sacred writing", "doctrine"],
    "religious clothing": ["traditional attire", "cultural dress", "faith-based garment", "sacred robe"],
    "practices": ["rituals", "customs", "traditions", "ceremonies", "observances"],
    "beliefs": ["doctrine", "faith", "ideology", "convictions", "tenets"],
    "trust": ["believe", "rely on", "have faith in", "count on", "depend on"],
    "avoid": ["stay away from", "shun", "ignore", "exclude", "ostracize"],
    "outdated": ["obsolete", "irrelevant", "antiquated", "passé", "old"],
    "strange": ["peculiar", "odd", "unusual", "bizarre", "foreign"],
    "causing problems": ["creating issues", "making trouble", "stirring conflict", "fueling unrest"],
    "ruining society": ["destroying culture", "corrupting values", "undermining civilization", "poisoning community"],
    "not funny": ["unamusing", "lame", "dull", "boring", "tasteless"],
    "incorrect": ["wrong", "erroneous", "inaccurate", "faulty", "flawed"],
    "unstable": ["unreliable", "intermittent", "patchy", "weak", "failing"],
    "slow": ["lagging", "sluggish", "delayed", "unresponsive", "tardy"],
    "frustrated": ["annoyed", "irritated", "exasperated", "fed up", "displeased"],
    "awful": ["terrible", "dreadful", "appalling", "atrocious", "deplorable"],
    "attitude": ["behavior", "demeanor", "conduct", "manner", "disposition"],
}

CONVERSATIONAL_PREFIXES = [
    "I heard someone say,",
    "A person told me,",
    "Someone posted online that",
    "I overheard,",
    "A message said,",
    "In the comments I read,",
    "This person wrote,",
    "At work someone said,",
    "In class I heard,",
    "My neighbor claims that",
    "Online I saw,",
    "A relative mentioned that",
    "In the group chat someone said,",
    "On social media I read,",
    "A coworker stated that",
    "During the discussion someone argued,",
    "In the forum post it said,",
    "A classmate mentioned,",
    "On the news comments someone wrote,",
    "I saw a message saying,",
]

CONVERSATIONAL_SUFFIXES = [
    "and I think that's unacceptable.",
    "which made me really uncomfortable.",
    "and honestly it's disturbing.",
    "which shows their ignorance.",
    "and people need to call this out.",
    "which is exactly why we need awareness.",
    "and this kind of talk needs to stop.",
    "which reflects deep prejudice.",
    "and it really upset the whole group.",
    "which nobody should have to hear.",
]

PASSIVE_VOICE_PATTERNS = [
    ("are", "are being"),
    ("is", "is being"),
    ("should", "should be"),
    ("need to", "need to be"),
    ("must", "must be"),
]

QUESTION_FORMS = [
    "Why do {subject} always {verb}?",
    "Don't you think {subject} are {adjective}?",
    "How can anyone defend {subject}?",
    "Isn't it obvious that {subject} {verb}?",
    "Why are {subject} allowed to {verb}?",
    "Who told {subject} they could {verb}?",
    "When will {subject} stop {verb}?",
    "What makes {subject} think they can {verb}?",
]

TYPO_MAP = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7',
}

SUBJECTS = {
    "ethnicity/race": ["people from your country", "your kind", "your people", "folks like you", "your community", "immigrants like you", "your ethnicity", "your background"],
    "gender/sexual": ["women like you", "people like you", "your type", "your kind", "girls", "men like you", "your gender", "your identity"],
    "religion": ["your faith group", "people of your religion", "your kind", "believers like you", "your community", "your faith", "practitioners like you", "your religious group"],
}

VERBS = {
    "ethnicity/race": ["cause problems", "ruin everything", "act so differently", "refuse to integrate", "bring crime", "take our jobs", "cause trouble", "act superior"],
    "gender/sexual": ["try to lead", "act so emotional", "push their agenda", "demand special treatment", "complain so much", "act weak", "cause drama", "ruin everything"],
    "religion": ["cause conflict", "push their beliefs", "refuse to adapt", "demand special rules", "spread hatred", "cause division", "act superior", "ruin society"],
}

ADJECTIVES = {
    "ethnicity/race": ["backward", "dangerous", "lazy", "violent", "untrustworthy", "inferior", "strange", "unwelcome"],
    "gender/sexual": ["weak", "emotional", "unnatural", "pathetic", "disgusting", "incapable", "weird", "inferior"],
    "religion": ["dangerous", "brainwashed", "backward", "violent", "intolerant", "fake", "strange", "divisive"],
}

CHALLENGING_NOT_CB = [
    # Contains "toxic" words but clearly neutral context
    ("I hate waiting in long lines at the grocery store.", "not_cyberbullying"),
    ("That movie was absolutely terrible and a waste of money.", "not_cyberbullying"),
    ("The traffic today is so stupid, I've been stuck for an hour.", "not_cyberbullying"),
    ("I can't stand the taste of pineapple on pizza.", "not_cyberbullying"),
    ("This software is completely dumb and hard to use.", "not_cyberbullying"),
    ("The weather is disgusting today, so humid and sticky.", "not_cyberbullying"),
    ("I hate how slow this computer is getting.", "not_cyberbullying"),
    ("That was a pathetic performance by the team last night.", "not_cyberbullying"),
    ("The service at that restaurant was awful and rude.", "not_cyberbullying"),
    ("I think horror movies are stupid and not scary at all.", "not_cyberbullying"),
    ("The WiFi connection here is weak and keeps dropping.", "not_cyberbullying"),
    ("I can't stand people who chew loudly in public.", "not_cyberbullying"),
    ("This book is so boring I can't finish it.", "not_cyberbullying"),
    ("The professor's lecture was confusing and poorly structured.", "not_cyberbullying"),
    ("I hate when my phone battery dies halfway through the day.", "not_cyberbullying"),
    ("That joke was offensive to no one, just not funny.", "not_cyberbullying"),
    ("The food was cold and the waiter was inattentive.", "not_cyberbullying"),
    ("I think pineapple on pizza is a crime against food.", "not_cyberbullying"),
    ("This game is so unfair and badly designed.", "not_cyberbullying"),
    ("I can't believe how expensive everything is getting.", "not_cyberbullying"),
    ("The bus was late again, this is so annoying.", "not_cyberbullying"),
    ("My neighbor's dog barks all night and I can't sleep.", "not_cyberbullying"),
    ("The printer is broken and I have a deadline tomorrow.", "not_cyberbullying"),
    ("I think this policy is completely wrong and needs review.", "not_cyberbullying"),
    ("The presentation was terrible and lacked any real data.", "not_cyberbullying"),
    ("I disagree strongly with that interpretation of the law.", "not_cyberbullying"),
    ("This coffee tastes awful, like burnt rubber.", "not_cyberbullying"),
    ("The meeting was a waste of time, nothing got decided.", "not_cyberbullying"),
    ("I can't stand this heat, it's unbearable outside.", "not_cyberbullying"),
    ("The customer support was useless and unhelpful.", "not_cyberbullying"),
    ("That movie plot made no sense whatsoever.", "not_cyberbullying"),
    ("I think your idea won't work for this project.", "not_cyberbullying"),
    ("The train was cancelled and I missed my appointment.", "not_cyberbullying"),
    ("I hate when people are late without any notice.", "not_cyberbullying"),
    ("This sofa is uncomfortable and badly made.", "not_cyberbullying"),
    ("The exam was unfair and covered topics we didn't study.", "not_cyberbullying"),
    ("I think that design choice is a mistake.", "not_cyberbullying"),
    ("The music was too loud and hurt my ears.", "not_cyberbullying"),
    ("I can't believe how rude some drivers are on the road.", "not_cyberbullying"),
    ("This app crashes constantly and it's frustrating.", "not_cyberbullying"),
    ("I think we should cancel the event due to low attendance.", "not_cyberbullying"),
    ("The hotel room was dirty and smelled bad.", "not_cyberbullying"),
    ("I hate dealing with paperwork and bureaucracy.", "not_cyberbullying"),
    ("The workshop was disorganized and ran over time.", "not_cyberbullying"),
    ("I think that approach is inefficient and outdated.", "not_cyberbullying"),
    ("The gym equipment is broken and needs repair.", "not_cyberbullying"),
    ("I can't stand this noise from the construction site.", "not_cyberbullying"),
    ("Your argument is weak and not well supported.", "not_cyberbullying"),
    ("I disagree with your choice of framework for this project.", "not_cyberbullying"),
    ("The lecture was boring and hard to follow.", "not_cyberbullying"),
    ("I think we need to reconsider our strategy here.", "not_cyberbullying"),
    ("The report contains several factual errors.", "not_cyberbullying"),
    ("I can't believe how slow the service was today.", "not_cyberbullying"),
]


def synonym_replace(text, p=0.3):
    """Replace words with synonyms with probability p."""
    words = text.split()
    new_words = []
    for w in words:
        clean = w.lower().strip(",.!?;:'\"")
        if clean in SYNONYMS and random.random() < p:
            replacement = random.choice(SYNONYMS[clean])
            # Preserve capitalization
            if w[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            if w[-1] in ",.!?;:'\"":
                replacement += w[-1]
            new_words.append(replacement)
        else:
            new_words.append(w)
    return " ".join(new_words)


def add_conversational_context(text, p=0.4):
    """Wrap text in conversational framing."""
    if random.random() < p:
        prefix = random.choice(CONVERSATIONAL_PREFIXES)
        if random.random() < 0.5:
            suffix = random.choice(CONVERSATIONAL_SUFFIXES)
            return f"{prefix} \"{text}\" {suffix}"
        else:
            return f"{prefix} \"{text}\""
    return text


def restructure_sentence(text, p=0.3):
    """Change sentence structure (active/passive, reorder)."""
    if random.random() > p:
        return text
    # Simple restructuring patterns
    patterns = [
        ("You ", "People like you "),
        ("Your ", "The "),
        ("We don't want ", "Nobody wants "),
        ("Stop ", "You need to stop "),
        ("Go back ", "Why don't you go back "),
        ("I hate ", "I can't stand "),
    ]
    for old, new in patterns:
        if text.startswith(old):
            return new + text[len(old):]
    # Try swapping clauses
    if ", " in text:
        parts = text.split(", ", 1)
        if len(parts) == 2 and random.random() < 0.5:
            return f"{parts[1].capitalize()}, {parts[0].lower()}"
    return text


def make_question(text, label, p=0.2):
    """Convert statement to rhetorical question."""
    if random.random() > p or label == "not_cyberbullying":
        return text
    subjects = SUBJECTS.get(label, ["people like you"])
    verbs = VERBS.get(label, ["cause problems"])
    adjectives = ADJECTIVES.get(label, ["bad"])
    template = random.choice(QUESTION_FORMS)
    return template.format(
        subject=random.choice(subjects),
        verb=random.choice(verbs),
        adjective=random.choice(adjectives)
    )


def inject_typos(text, p=0.15):
    """Inject subtle typo-like noise."""
    if random.random() > p:
        return text
    words = text.split()
    idx = random.randint(0, len(words) - 1)
    word = words[idx]
    if len(word) > 3:
        char_idx = random.randint(1, len(word) - 2)
        if word[char_idx].lower() in TYPO_MAP:
            word = word[:char_idx] + TYPO_MAP[word[char_idx].lower()] + word[char_idx+1:]
            words[idx] = word
    return " ".join(words)


def vary_intensity(text, label, p=0.25):
    """Make hate speech more implicit/subtle."""
    if random.random() > p or label == "not_cyberbullying":
        return text
    # Replace explicit slurs/toxic words with euphemisms or implications
    replacements = [
        ("hate", "don't appreciate"),
        ("shouldn't be allowed", "don't really fit"),
        ("go back", "might be happier elsewhere"),
        ("nobody wants", "not everyone is comfortable with"),
        ("completely inferior", "quite different from"),
        ("are dumb", "struggle with this"),
        ("disgusting", "hard to understand"),
        ("pathetic", "concerning"),
        ("violent cult", "controversial group"),
        ("brainwashed", "strongly influenced"),
        ("fake", "different from what I know"),
        ("stupid", "questionable"),
        ("backward", "traditional"),
        ("ruining", "changing"),
        ("dangerous", "concerning"),
        ("weak", "different"),
        ("unnatural", "uncommon"),
    ]
    for old, new in replacements:
        if old in text.lower():
            # Case-insensitive replacement
            import re
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
            break
    return text


def generate_from_template(template, label, num_variations=1):
    """Generate multiple variations from a single template."""
    results = []
    for _ in range(num_variations):
        text = template
        text = synonym_replace(text, p=0.4)
        text = restructure_sentence(text, p=0.3)
        text = add_conversational_context(text, p=0.35)
        text = make_question(text, label, p=0.15)
        text = vary_intensity(text, label, p=0.2)
        text = inject_typos(text, p=0.1)
        results.append(text)
    return results


def generate_class_samples(templates, label, target_count):
    """Generate target_count synthetic samples for a class."""
    samples = []
    samples_per_template = max(1, target_count // len(templates))
    extra = target_count - (samples_per_template * len(templates))
    
    for i, template in enumerate(templates):
        count = samples_per_template + (1 if i < extra else 0)
        variations = generate_from_template(template, label, num_variations=count)
        for v in variations:
            samples.append({
                "text": v,
                "label": label,
                "source_type": "synthetic",
                "original_template": template,
                "generation_strategy": "template_variation",
                "created_at": datetime.now().isoformat(),
            })
    
    # If still short, generate more with random combinations
    while len(samples) < target_count:
        template = random.choice(templates)
        variations = generate_from_template(template, label, num_variations=1)
        for v in variations:
            samples.append({
                "text": v,
                "label": label,
                "source_type": "synthetic",
                "original_template": template,
                "generation_strategy": "random_combination",
                "created_at": datetime.now().isoformat(),
            })
    
    return samples[:target_count]


def generate_challenging_not_cb(target_count):
    """Generate challenging not_cyberbullying samples."""
    samples = []
    
    # Use the base challenging examples
    for text, label in CHALLENGING_NOT_CB:
        # Generate variations
        for _ in range(3):  # 3 variations per base
            v = synonym_replace(text, p=0.2)
            v = inject_typos(v, p=0.05)
            samples.append({
                "text": v,
                "label": label,
                "source_type": "synthetic",
                "original_template": text,
                "generation_strategy": "challenging_neutral",
                "created_at": datetime.now().isoformat(),
            })
    
    # Add more variations by changing context
    context_variations = [
        "I really {verb} when {situation}.",
        "The {noun} was so {adj} today.",
        "Can you believe how {adj} this {noun} is?",
        "I think the {noun} needs to be {verb}.",
        "Why is the {noun} always so {adj}?",
        "This {noun} is completely {adj} and I am {emotion}.",
    ]
    
    nouns = ["service", "food", "weather", "traffic", "internet", "movie", "game", "app", "phone", "computer", "meeting", "lecture", "workshop", "presentation", "performance", "system", "process", "policy", "design", "approach"]
    verbs = ["hate", "dislike", "can't stand", "detest", "despise", "loathe"]
    adjs = ["terrible", "awful", "pathetic", "horrible", "dreadful", "appalling", "atrocious", "disgusting", "stupid", "dumb", "weak", "slow", "broken", "useless", "annoying", "frustrating", "boring", "confusing", "unfair", "rude", "cold", "late", "expensive", "loud"]
    emotions = ["frustrated", "annoyed", "disappointed", "irritated", "fed up", "exasperated", "angry", "upset", "bored", "confused", "concerned", "worried", "stressed", "tired", "sick"]
    
    while len(samples) < target_count:
        template = random.choice(context_variations)
        text = template.format(
            verb=random.choice(verbs),
            situation=random.choice([
                "the bus is late", "the WiFi drops", "prices go up", "the line is long",
                "the food is cold", "the meeting runs over", "the app crashes",
                "the traffic gets worse", "it rains on weekends", "the printer breaks"
            ]),
            noun=random.choice(nouns),
            adj=random.choice(adjs),
            emotion=random.choice(emotions)
        )
        samples.append({
            "text": text,
            "label": "not_cyberbullying",
            "source_type": "synthetic",
            "original_template": template,
            "generation_strategy": "context_variation",
            "created_at": datetime.now().isoformat(),
        })
    
    return samples[:target_count]


def main():
    print("=" * 60)
    print("Generating Synthetic Detection Dataset")
    print("=" * 60)
    
    # Generate cyberbullying classes
    ethnicity_samples = generate_class_samples(ETHNICITY_TEMPLATES, "ethnicity/race", 500)
    gender_samples = generate_class_samples(GENDER_TEMPLATES, "gender/sexual", 500)
    religion_samples = generate_class_samples(RELIGION_TEMPLATES, "religion", 500)
    not_cb_samples = generate_challenging_not_cb(500)
    
    all_samples = ethnicity_samples + gender_samples + religion_samples + not_cb_samples
    random.shuffle(all_samples)
    
    # Verify counts
    counts = {}
    for s in all_samples:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    print(f"\nGenerated {len(all_samples)} synthetic samples:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    
    # Save CSV
    csv_path = os.path.join("data", "synthetic", "detection_synthetic.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source_type", "original_template", "generation_strategy", "created_at"])
        writer.writeheader()
        writer.writerows(all_samples)
    print(f"\n[Saved] CSV -> {csv_path}")
    
    # Save JSON
    json_path = os.path.join("data", "synthetic", "detection_synthetic.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_samples": len(all_samples),
                "class_distribution": counts,
                "generation_date": datetime.now().isoformat(),
                "source": "template-based augmentation from failure examples",
                "strategies": ["synonym_replacement", "conversational_context", "restructuring", "question_forms", "intensity_variation", "typo_injection", "challenging_neutral"],
            },
            "samples": all_samples
        }, f, indent=2, ensure_ascii=False)
    print(f"[Saved] JSON -> {json_path}")
    
    # Also save a simple merged dataset
    import pandas as pd
    original_df = pd.read_csv("cb_multi_labeled_balanced.csv")
    synthetic_df = pd.DataFrame([{"text": s["text"], "label": s["label"]} for s in all_samples])
    merged_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    merged_path = os.path.join("data", "synthetic", "cb_enhanced.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"[Saved] Merged dataset -> {merged_path}")
    print(f"  Original: {len(original_df)} | Synthetic: {len(synthetic_df)} | Total: {len(merged_df)}")
    
    return all_samples


if __name__ == "__main__":
    main()
