"""
Generate synthetic reply generation dataset.
Outputs: data/synthetic/reply_synthetic.csv + .json
300 pairs: 100 per cyberbullying class
"""
import json
import random
import csv
import os
from datetime import datetime

random.seed(44)

# ── Cyberbullying inputs paired with safe replies ──────────────────

REPLY_PAIRS = [
    # ethnicity/race
    ("Go back to where you came from, nobody wants you here.", "You belong here just as much as anyone else. Your presence matters and this community is better with you in it."),
    ("People of your race are always ruining this country.", "I'm sorry you had to read that. Your heritage is something to be proud of, and you deserve respect."),
    ("Your culture is completely inferior to ours.", "Every culture brings unique beauty and wisdom to the world. Your identity is valuable and worthy of celebration."),
    ("You people are all the same.", "You are an individual with your own thoughts, dreams, and worth. No stereotype can define who you are."),
    ("Your accent makes you sound uneducated.", "Your accent tells a beautiful story of your journey and heritage. Your voice deserves to be heard."),
    ("No wonder you failed, people like you never succeed.", "One person's cruel words don't determine your potential. You are capable of amazing things."),
    ("Why are people from your ethnicity always causing problems?", "Stereotypes like that are unfair and hurtful. You are not responsible for other people's prejudice."),
    ("Every post from your race is annoying.", "Your perspective matters. Don't let ignorance silence your voice or dim your light."),
    ("Your community should stay off social media.", "You have every right to share your voice and experiences. This space belongs to everyone."),
    ("Typical behavior from someone of your race.", "You are uniquely you, not a stereotype. Your actions and character speak for themselves."),
    ("Go back to your own country, nobody wants you here.", "This is your home too. You deserve safety, belonging, and respect wherever you are."),
    ("People from your race always ruin everything online.", "I'm sorry you experienced that hate. Your contributions online are valuable and welcome."),
    ("Your ethnicity is the reason you act so dumb.", "Intelligence has nothing to do with ethnicity. You are capable, worthy, and deserving of respect."),
    ("No wonder you failed, people like you never succeed.", "Failure is temporary and learning is lifelong. Your potential is unlimited, regardless of what anyone says."),
    ("Your accent makes you sound uneducated.", "Multilingualism and diverse accents are strengths. Your communication skills are valuable assets."),
    ("Every post from your race is annoying.", "Your voice matters. Please don't let hate discourage you from sharing your thoughts and experiences."),
    ("You people are all the same.", "You are a unique individual with your own story, strengths, and dreams. No one can put you in a box."),
    ("Why are people from your ethnicity always causing problems?", "That question reflects their prejudice, not any truth about you. You are not defined by their ignorance."),
    ("Your community should stay off social media.", "Social media is for everyone. Your community's voices deserve to be heard and celebrated."),
    ("Typical behavior from someone of your race.", "Stereotypes limit the person using them, not you. You define yourself through your actions and character."),
    ("Go back to your own country.", "You have every right to be here. Your presence enriches this community in ways that narrow minds cannot see."),
    ("People from your race always ruin everything online.", "Hate online says everything about the sender and nothing about you. Keep sharing your authentic self."),
    ("Your ethnicity is the reason you act so dumb.", "Intelligence is individual, not ethnic. Your mind is sharp, your thoughts matter, and your growth is your own."),
    ("No wonder you failed, people like you never succeed.", "Success looks different for everyone, and your journey is uniquely yours. One person's hate cannot limit your future."),
    ("Your accent makes you sound uneducated.", "Speaking multiple languages or having an accent shows cognitive flexibility and cultural richness. Be proud."),
    ("Every post from your race is annoying.", "Diversity of thought makes communities stronger. Your perspective adds value that this space needs."),
    ("You people are all the same.", "You are irreplaceable. Your individual experiences, thoughts, and feelings make you one of a kind."),
    ("Why are people from your ethnicity always causing problems?", "That unfair generalization says more about their bias than about anyone from your background. You are not responsible."),
    ("Your community should stay off social media.", "Every community deserves representation. Your voice online helps others feel seen and less alone."),
    ("Typical behavior from someone of your race.", "Labels don't define you. Your kindness, resilience, and unique spirit are what truly matter."),
    ("Go back to where your race belongs.", "You belong wherever you choose to be. No one has the right to decide where you do or don't belong."),
    ("People from your ethnicity are always so backward.", "Cultural evolution happens everywhere. Your heritage contains centuries of wisdom and innovation."),
    ("Nobody likes your kind here.", "You are liked, valued, and wanted. One person's hate cannot speak for an entire community."),
    ("No wonder you failed, people from your race are dumb.", "Racist stereotypes have no basis in reality. Your capabilities are limitless and your future is bright."),
    ("Your people should stay separate from us.", "Integration and mutual respect make societies stronger. You deserve to be part of every space you enter."),

    # gender/sexual
    ("Go back to the kitchen and leave the real work to the men.", "Your skills and ambitions are valid regardless of gender. You belong in every room you enter."),
    ("Women are way too emotional to ever hold a leadership position.", "Emotional intelligence is a leadership strength. Your ability to lead with empathy is an asset, not a weakness."),
    ("People with your sexual orientation are completely unnatural and wrong.", "You are perfect exactly as you are. Your identity is natural, valid, and worthy of love and respect."),
    ("Girls can't handle technical work.", "Women have always excelled in technical fields. Your brain is capable of anything you set your mind to."),
    ("Boys who cry are weak and pathetic.", "Expressing emotions takes courage and strength. Your vulnerability is a sign of bravery, not weakness."),
    ("Nobody takes women gamers seriously.", "Women belong in gaming just as much as anyone. Your skills speak for themselves, and your passion is valid."),
    ("You act too feminine to be respected.", "Femininity and respect are not mutually exclusive. You deserve respect exactly as you present yourself."),
    ("Men like you are completely useless.", "Your worth is not determined by outdated stereotypes. You have unique strengths that the world needs."),
    ("Your gender identity is just for attention.", "Your identity is real, valid, and deeply personal. No one else gets to define who you are."),
    ("Stop pretending to be something you're not.", "Living authentically is brave and beautiful. You are not pretending—you are becoming more fully yourself."),
    ("You'll never fit in because of who you are.", "The right people will love and accept you exactly as you are. You don't need to fit into spaces that don't deserve you."),
    ("Only idiots would support someone like you.", "Those who truly see you will stand by you. Their support is worth more than the noise of ignorance."),
    ("People like you shouldn't be allowed to marry or have families.", "You deserve love, family, and happiness just like everyone else. Your relationships are beautiful and valid."),
    ("This is a man's field, girls shouldn't even bother trying.", "Fields don't have genders, only people with passion and skill. You have both, and you belong here."),
    ("Your lifestyle choice is disgusting and shouldn't be shown in public.", "Living authentically is not a choice, it's a right. Your presence in public spaces is valid and important."),
    ("You sound like a typical emotional woman who can't handle logic.", "Emotion and logic coexist in all humans. Your perspective is rational, valuable, and complete."),
    ("You are a man, start acting like one instead of crying.", "There is no single way to be a man. Your emotions make you human, and your tears show your strength."),
    ("Nobody respects a woman who tries to speak up in this forum.", "Your voice matters. Speaking up takes courage, and respect is earned by those brave enough to use their voice."),
    ("Girls can't understand technology properly.", "Understanding has no gender. Your technical abilities are real, valuable, and deserving of recognition."),
    ("Boys who cry are weak and pathetic.", "Emotional expression is a sign of maturity and strength. Your feelings are valid and your tears are brave."),
    ("Nobody takes women gamers seriously.", "Serious gamers come in all genders. Your passion and skill make you a gamer, period."),
    ("You act too feminine to be respected.", "Respect should never depend on how masculine or feminine you appear. You deserve respect as you are."),
    ("Men like you are completely useless.", "You are not useless. Everyone has purpose and value, including you, exactly as you are."),
    ("Your gender identity is just for attention.", "Your truth is not performative. You are living authentically, and that deserves admiration, not dismissal."),
    ("Stop pretending to be something you're not.", "You are being your truest self, and that is powerful. Authenticity is never pretending."),
    ("You'll never fit in because of who you are.", "You will find your people. The world is full of kind hearts who will celebrate you exactly as you are."),
    ("Only idiots would support someone like you.", "Support from those who truly see you is what matters. Their love outweighs the ignorance of strangers."),
    ("People like you shouldn't be allowed to marry or have families.", "Love is love, and family is family. You deserve every happiness that others take for granted."),
    ("This is a man's field, girls shouldn't even bother trying.", "Every field benefits from diverse perspectives. Your presence makes the industry better and more innovative."),
    ("Your lifestyle choice is disgusting and shouldn't be shown in public.", "Living openly and honestly is courageous. Your visibility helps others feel less alone."),
    ("You sound like a typical emotional woman who can't handle logic.", "Empathy and logic are not opposites. Your emotional intelligence enhances your reasoning, not diminishes it."),
    ("You are a man, start acting like one instead of crying.", "Real strength includes vulnerability. Your ability to feel deeply is one of your greatest powers."),
    ("Nobody respects a woman who tries to speak up in this forum.", "History is made by women who spoke up despite opposition. Your voice is paving the way for others."),

    # religion
    ("Your religion is a violent cult that needs to be banned.", "I'm sorry you faced such ignorance. Your faith is personal and meaningful, and you deserve respect."),
    ("All people who practice your faith are completely brainwashed.", "Your beliefs come from genuine reflection and tradition. No one has the right to delegitimize your spirituality."),
    ("Your religious beliefs are completely backwards and stupid.", "Faith evolves differently for everyone. Your beliefs deserve the same respect as any other worldview."),
    ("Your holy book is full of lies and promotes hatred.", "Sacred texts are interpreted in countless ways. Your practice of faith is about love and peace, not hate."),
    ("Go pray instead of embarrassing yourself online.", "Your faith and your online presence are both valid. You have as much right to speak as anyone else."),
    ("Your religion should be banned everywhere.", "Religious freedom is a fundamental right. Your practice harms no one and deserves protection, not persecution."),
    ("Only fools follow your religious practices.", "Wisdom takes many forms across cultures. Your spiritual practice reflects deep values and connection."),
    ("Your community is always causing trouble.", "A few voices don't represent an entire community. Your faith community likely does much good in the world."),
    ("People from your religion don't belong here.", "You belong wherever you are. Your faith is part of who you are, and you deserve to be accepted fully."),
    ("I can't believe anyone still follows that religion.", "Faith persists because it provides meaning and hope to billions. Your beliefs are valid and time-honored."),
    ("Your religion is the reason society has problems.", "Complex societal issues have many causes. Blaming an entire religion ignores the good its followers do daily."),
    ("People from your faith are so brainwashed.", "Devotion is not brainwashing. Your faith is a conscious choice that brings you peace and purpose."),
    ("Nobody trusts followers of your religion.", "Trust is built individually. You are trustworthy because of your character, not despite your faith."),
    ("Your beliefs are completely stupid.", "Beliefs that provide comfort, community, and moral guidance are not stupid. They are deeply human and valuable."),
    ("Go pray instead of embarrassing yourself online.", "Prayer and public discourse aren't mutually exclusive. Your faith strengthens your voice, not weakens it."),
    ("Your religion should be banned everywhere.", "Banning religions has historically caused suffering. Your peaceful practice deserves protection and respect."),
    ("Only fools follow your religious practices.", "Rituals connect people to tradition and meaning. Your practices reflect wisdom passed down through generations."),
    ("Your community is always causing trouble.", "Every community has diverse members. Judge individuals by their actions, not groups by stereotypes."),
    ("People from your religion don't belong here.", "Diversity of belief enriches society. Your presence makes this community more compassionate and understanding."),
    ("I can't believe anyone still follows that religion.", "Ancient wisdom still resonates today. Your faith offers timeless guidance that continues to help millions."),
    ("Your religion is the reason society has problems.", "Society's challenges are complex and multi-causal. Your faith community likely contributes solutions, not problems."),
    ("People from your faith are so brainwashed.", "Faith is a gift for many, not a condition. Your spiritual journey is valid and worthy of respect."),
    ("Nobody trusts followers of your religion.", "Trust is earned through consistent integrity. Show your character, and let it speak louder than prejudice."),
    ("Your beliefs are completely stupid.", "Beliefs that sustain people through hardship are profound, not stupid. Your faith is a source of genuine strength."),
    ("Go pray instead of embarrassing yourself online.", "Your spiritual life and public voice both matter. You don't need to choose between them."),
    ("Your religion should be banned everywhere.", "Religious pluralism makes democracies stronger. Your right to believe is as important as anyone else's."),
    ("Only fools follow your religious practices.", "Rituals provide structure, meaning, and community. Your practices reflect intelligence about what humans need."),
    ("Your community is always causing trouble.", "Most communities strive to do good. Your faith likely inspires charity, compassion, and service to others."),
    ("People from your religion don't belong here.", "Everyone belongs somewhere. Your faith is part of your identity, and this space is better for having you."),
    ("I can't believe anyone still follows that religion.", "Faith endures because it answers deep human needs. Your beliefs are a legitimate and meaningful part of your life."),
    ("Your faith makes you dangerous.", "Your faith likely teaches peace and compassion. You are not dangerous—you are a person of conscience and care."),
    ("Everyone from your religion thinks the same way.", "Religious communities are incredibly diverse in thought. Your individual perspective is unique and valuable."),
    ("Your religious practices are stupid.", "Practices that bring peace and connection are wise, not stupid. Your traditions hold deep meaning."),
    ("No one wants your religion around.", "Many people appreciate learning about different faiths. Your religion adds richness to human diversity."),
]

# Ensure exactly 300 pairs
print(f"Total reply pairs defined: {len(REPLY_PAIRS)}")

# If more than 300, sample; if less, duplicate with minor variations
if len(REPLY_PAIRS) > 300:
    REPLY_PAIRS = random.sample(REPLY_PAIRS, 300)
elif len(REPLY_PAIRS) < 300:
    while len(REPLY_PAIRS) < 300:
        pair = random.choice(REPLY_PAIRS)
        # Create a minor variation
        text, reply = pair
        variations = [
            (text, reply + " Remember, you are not alone."),
            (text, reply + " Please reach out if you need support."),
            (text, reply + " Stay strong and true to yourself."),
            (text, reply + " You matter more than you know."),
        ]
        REPLY_PAIRS.append(random.choice(variations))

random.shuffle(REPLY_PAIRS)

# Categorize by class
categorized = {"ethnicity/race": [], "gender/sexual": [], "religion": []}
for text, reply in REPLY_PAIRS:
    # Simple heuristic categorization based on keywords
    t_lower = text.lower()
    if any(w in t_lower for w in ["race", "ethnicity", "accent", "country", "heritage", "culture"]):
        categorized["ethnicity/race"].append((text, reply))
    elif any(w in t_lower for w in ["gender", "women", "men", "feminine", "sexual", "gay", "lgbt", "trans"]):
        categorized["gender/sexual"].append((text, reply))
    elif any(w in t_lower for w in ["religion", "faith", "pray", "holy", "god", "belief", "religious"]):
        categorized["religion"].append((text, reply))
    else:
        # Default to gender/sexual for ambiguous cases
        categorized["gender/sexual"].append((text, reply))

# Build final 300 with balanced classes (~100 each)
final_pairs = []
target_per_class = 100
for label, pairs in categorized.items():
    # Take up to target_per_class, fill with random if short
    selected = pairs[:target_per_class]
    while len(selected) < target_per_class:
        selected.append(random.choice(pairs))
    for text, reply in selected:
        final_pairs.append({
            "cyberbullying_text": text,
            "safe_reply": reply,
            "label": label,
            "source_type": "synthetic",
            "generation_strategy": "human_curated_template",
            "created_at": datetime.now().isoformat(),
        })

random.shuffle(final_pairs)

print(f"Final dataset: {len(final_pairs)} pairs")
print("Class distribution:")
for label in ["ethnicity/race", "gender/sexual", "religion"]:
    count = sum(1 for p in final_pairs if p["label"] == label)
    print(f"  {label}: {count}")

# Save CSV
csv_path = os.path.join("data", "synthetic", "reply_synthetic.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["cyberbullying_text", "safe_reply", "label", "source_type", "generation_strategy", "created_at"])
    writer.writeheader()
    writer.writerows(final_pairs)
print(f"\n[Saved] CSV -> {csv_path}")

# Save JSON
json_path = os.path.join("data", "synthetic", "reply_synthetic.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "total_pairs": len(final_pairs),
            "class_distribution": {label: sum(1 for p in final_pairs if p["label"] == label) for label in ["ethnicity/race", "gender/sexual", "religion"]},
            "generation_date": datetime.now().isoformat(),
            "source": "human-curated synthetic pairs",
            "description": "Cyberbullying text paired with empathetic, non-confrontational safe reply"
        },
        "pairs": final_pairs
    }, f, indent=2, ensure_ascii=False)
print(f"[Saved] JSON -> {json_path}")
