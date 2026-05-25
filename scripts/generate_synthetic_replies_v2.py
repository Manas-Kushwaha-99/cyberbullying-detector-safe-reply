"""
Generate 1000 synthetic de-escalatory reply pairs (CORRECTED).
Reply is directed AT THE AGGRESSOR to de-escalate, not comforting the victim.
Outputs: data/synthetic/reply_synthetic_v2.csv + .json
"""
import json
import random
import csv
import os
from datetime import datetime

random.seed(45)

# ── Base templates: (aggressor_text, de_escalatory_reply_to_aggressor) ──

BASE_PAIRS = [
    # ethnicity/race
    ("Go back to where you came from, nobody wants you here.", "Everyone has the right to be here. Please treat others with the same respect you would want for yourself."),
    ("People of your race are always ruining this country.", "That's a harmful stereotype with no basis in fact. Every group contributes to our society in meaningful ways."),
    ("Your culture is completely inferior to ours.", "No culture is superior or inferior to another. Each has its own unique value and contributions to humanity."),
    ("You people are all the same.", "That's a stereotype, not a truth. Every person is an individual with their own character and abilities."),
    ("Your accent makes you sound uneducated.", "Accents reflect linguistic diversity and cultural heritage, not education. Please be respectful."),
    ("No wonder you failed, people like you never succeed.", "Success comes in many forms and isn't determined by background. Let's focus on supporting each other instead."),
    ("Why are people from your ethnicity always causing problems?", "That's an unfair generalization. Individuals are responsible for their own actions, not entire communities."),
    ("Every post from your race is annoying.", "If you don't like someone's content, you can simply scroll past. There's no need for disrespect."),
    ("Your community should stay off social media.", "Social media is for everyone. If you disagree with content, engage respectfully or move on."),
    ("Typical behavior from someone of your race.", "Attributing behavior to an entire race is prejudice. Please judge people as individuals, not stereotypes."),
    ("Go back to your own country, nobody wants you here.", "This country belongs to all its citizens and residents. Xenophobia has no place in civil discourse."),
    ("People from your race always ruin everything online.", "Blaming an entire race for online issues is prejudice. Let's address problems without attacking groups."),
    ("Your ethnicity is the reason you act so dumb.", "Intelligence is individual, not ethnic. Such comments reflect poorly on the speaker, not the target."),
    ("No wonder you failed, people like you never succeed.", "Failure is part of learning for everyone. Success isn't determined by identity but by effort and opportunity."),
    ("Your accent makes you sound uneducated.", "Speaking multiple languages or having an accent demonstrates cognitive flexibility, not lack of education."),
    ("Every post from your race is annoying.", "If content isn't for you, simply don't engage. Attacking an entire group is unnecessary and harmful."),
    ("You people are all the same.", "Reducing individuals to group stereotypes prevents meaningful connection. Everyone deserves to be seen as themselves."),
    ("Why are people from your ethnicity always causing problems?", "Generalizing negative behavior to an ethnicity is factually wrong and morally harmful. Please reconsider."),
    ("Your community should stay off social media.", "Exclusion based on community membership violates the principles of open discourse. Everyone deserves a voice."),
    ("Typical behavior from someone of your race.", "There is no 'typical behavior' for any race. Such statements reveal bias, not truth."),
    ("Go back to where your race belongs.", "People belong wherever they choose to live and contribute. Migration is a fundamental human right."),
    ("People from your ethnicity are always so backward.", "Labeling entire ethnicities as 'backward' is ethnocentric and ignores the complexity of cultural development."),
    ("Nobody likes your kind here.", "'Your kind' is a divisive phrase. We're all humans sharing this space, and mutual respect benefits everyone."),
    ("No wonder you failed, people from your race are dumb.", "Racist stereotypes about intelligence have been scientifically debunked repeatedly. Please educate yourself."),
    ("Your people should stay separate from us.", "Segregation has never led to peace or progress. Integration and mutual understanding build stronger societies."),
    ("People from your race always ruin everything online.", "Online spaces are shared by all. Constructive criticism addresses behavior, not identity."),
    ("Your ethnicity is the reason you act so dumb.", "Behavior is shaped by individual choices and circumstances, not ethnicity. That assumption is both false and harmful."),
    ("No wonder you failed, people like you never succeed.", "Attributing individual setbacks to group identity is prejudice. Success is multifaceted and deeply personal."),
    ("Your accent makes you sound uneducated.", "Accents indicate multilingual ability and cultural background. They have no correlation with intelligence or education."),
    ("Every post from your race is annoying.", "Annoyance is subjective. Please practice tolerance or use platform tools to curate your experience respectfully."),
    ("You people are all the same.", "That statement erases individual humanity. Each person is unique, regardless of their background."),
    ("Why are people from your ethnicity always causing problems?", "Problems in any community have complex socioeconomic roots. Scapegoating ethnicities solves nothing."),
    ("Your community should stay off social media.", "Social media thrives on diverse perspectives. Silencing communities weakens democratic discourse."),
    ("Typical behavior from someone of your race.", "Attributing individual actions to racial characteristics is the definition of racial prejudice."),

    # gender/sexual
    ("Go back to the kitchen and leave the real work to the men.", "Gender doesn't determine capability or career suitability. Everyone deserves respect for their contributions."),
    ("Women are way too emotional to ever hold a leadership position.", "Emotional intelligence is a leadership asset, not a liability. Many great leaders lead with empathy."),
    ("People with your sexual orientation are completely unnatural and wrong.", "Sexual orientation is a natural variation of human diversity. Respect for others is never wrong."),
    ("Girls can't handle technical work.", "Technical ability depends on training and interest, not gender. Many women excel in technical fields."),
    ("Boys who cry are weak and pathetic.", "Emotional expression is a sign of strength and authenticity. Suppressing feelings causes real harm."),
    ("Nobody takes women gamers seriously.", "Women have always been part of gaming. Skill and passion define a gamer, not gender."),
    ("You act too feminine to be respected.", "Respect should never depend on gender expression. Everyone deserves dignity regardless of how they present."),
    ("Men like you are completely useless.", "Every person has intrinsic value. Reducing anyone to 'useless' based on gender is cruel and false."),
    ("Your gender identity is just for attention.", "Gender identity is deeply personal and real. Dismissing someone's identity shows a lack of empathy, not insight."),
    ("Stop pretending to be something you're not.", "Living authentically is not pretense. The only thing people should stop is judging others for being themselves."),
    ("You'll never fit in because of who you are.", "True communities accept people as they are. If someone doesn't fit your narrow mold, that says more about you than them."),
    ("Only idiots would support someone like you.", "Supporting people for being their authentic selves is compassionate, not idiotic. Prejudice is the real foolishness."),
    ("People like you shouldn't be allowed to marry or have families.", "Love and family are universal human rights. Who someone loves doesn't affect your life in any way."),
    ("This is a man's field, girls shouldn't even bother trying.", "No field has a gender. Talent and dedication are what matter, and they exist in all genders equally."),
    ("Your lifestyle choice is disgusting and shouldn't be shown in public.", "Being oneself isn't a 'lifestyle choice' to be hidden. Visibility saves lives and builds acceptance."),
    ("You sound like a typical emotional woman who can't handle logic.", "Emotion and logic coexist in all humans. Framing women as illogical is a tired stereotype without merit."),
    ("You are a man, start acting like one instead of crying.", "There is no single way to 'act like a man.' Emotional honesty is braver than conforming to outdated stereotypes."),
    ("Nobody respects a woman who tries to speak up in this forum.", "Speaking up requires courage. Respect should be given to those brave enough to use their voice."),
    ("Girls can't understand technology properly.", "Understanding technology depends on education and practice, not gender. Countless women are tech leaders."),
    ("Boys who cry are weak and pathetic.", "Emotional vulnerability is a strength. Mocking it perpetuates a toxic cycle that harms everyone."),
    ("Nobody takes women gamers seriously.", "Seriousness in gaming comes from skill and commitment, not gender. Gatekeeping helps no one."),
    ("You act too feminine to be respected.", "Respect should be unconditional. Gender expression is personal and harms no one."),
    ("Men like you are completely useless.", "Value isn't determined by conforming to gender roles. Everyone contributes in their own unique way."),
    ("Your gender identity is just for attention.", "For most people, gender identity is private and deeply held. Assuming it's performative reveals your bias, not their truth."),
    ("Stop pretending to be something you're not.", "The only pretense here is the idea that everyone should fit your expectations. Authenticity is not performance."),
    ("You'll never fit in because of who you are.", "Fitting into intolerant spaces isn't the goal. Building inclusive communities where everyone belongs is."),
    ("Only idiots would support someone like you.", "Support for marginalized people reflects moral courage. It takes no bravery to punch down."),
    ("People like you shouldn't be allowed to marry or have families.", "Denying people the right to love and build families is cruelty, not moral standing."),
    ("This is a man's field, girls shouldn't even bother trying.", "Innovation thrives on diversity. Excluding women from any field makes the field weaker, not stronger."),
    ("Your lifestyle choice is disgusting and shouldn't be shown in public.", "Loving openly and living authentically isn't 'disgusting.' Intolerance is what harms society."),
    ("You sound like a typical emotional woman who can't handle logic.", "This stereotype has been used for centuries to silence women. Emotion and reason are complementary, not opposites."),
    ("You are a man, start acting like one instead of crying.", "Real masculinity includes compassion and emotional range. Rigid gender roles hurt men most of all."),
    ("Nobody respects a woman who tries to speak up in this forum.", "History is full of women who changed the world by speaking up. Silence never led to progress."),

    # religion
    ("Your religion is a violent cult that needs to be banned.", "Labeling entire religions as violent ignores the peaceful majority. Religious freedom is a fundamental right."),
    ("All people who practice your faith are completely brainwashed.", "Devotion is not brainwashing. Billions find meaning, morality, and community through their faith."),
    ("Your religious beliefs are completely backwards and stupid.", "Beliefs that provide meaning to billions deserve respect, even if you don't share them."),
    ("Your holy book is full of lies and promotes hatred.", "Sacred texts are interpreted diversely by their followers. Most believers emphasize peace and compassion."),
    ("Go pray instead of embarrassing yourself online.", "Faith and public discourse aren't mutually exclusive. Believers have as much right to participate as anyone."),
    ("Your religion should be banned everywhere.", "Banning religions is a hallmark of authoritarianism. Religious pluralism strengthens democratic societies."),
    ("Only fools follow your religious practices.", "Rituals provide structure and meaning to billions. Dismissing them as foolish reveals ignorance, not wisdom."),
    ("Your community is always causing trouble.", "A few individuals don't represent an entire faith community. Most religious communities contribute positively to society."),
    ("People from your religion don't belong here.", "This space belongs to everyone, regardless of faith. Pluralism is a strength, not a weakness."),
    ("I can't believe anyone still follows that religion.", "Faith endures because it answers deep human needs for meaning, community, and moral guidance."),
    ("Your religion is the reason society has problems.", "Society's problems are complex and multifaceted. Scapegoating religions oversimplifies and divides."),
    ("People from your faith are so brainwashed.", "Strong faith is often the result of genuine experience and reflection, not manipulation."),
    ("Nobody trusts followers of your religion.", "Trust is built individually. Millions of people of all faiths are trusted friends, colleagues, and leaders."),
    ("Your beliefs are completely stupid.", "Beliefs that sustain people through hardship and guide moral behavior are far from stupid."),
    ("Go pray instead of embarrassing yourself online.", "Prayer and public expression are both valid. There's no need to mock someone's spiritual practice."),
    ("Your religion should be banned everywhere.", "History shows that banning religions leads to persecution and conflict, not peace."),
    ("Only fools follow your religious practices.", "Practices that connect people to tradition and meaning reflect wisdom about human needs, not foolishness."),
    ("Your community is always causing trouble.", "Every community has diverse members. Judge individuals by their actions, not groups by stereotypes."),
    ("People from your religion don't belong here.", "Exclusion based on faith contradicts principles of equality and human dignity. Everyone belongs."),
    ("I can't believe anyone still follows that religion.", "Ancient wisdom continues to resonate because it addresses timeless human questions about purpose and ethics."),
    ("Your religion is the reason society has problems.", "Blaming religion for complex social issues ignores economics, politics, and history. It's a convenient but false narrative."),
    ("People from your faith are so brainwashed.", "Critical thinking and faith coexist for many. Assuming believers haven't thought critically is itself uncritical."),
    ("Nobody trusts followers of your religion.", "Prejudice creates distrust, not faith itself. Many of the most trusted people in history were deeply religious."),
    ("Your beliefs are completely stupid.", "Beliefs that have sustained civilizations and inspired billions of acts of charity are profound, not stupid."),
    ("Go pray instead of embarrassing yourself online.", "Spiritual expression and online participation aren't mutually exclusive. Mockery says more about the mocker."),
    ("Your religion should be banned everywhere.", "Religious freedom is a cornerstone of liberal democracy. Banning faiths undermines everyone's liberty."),
    ("Only fools follow your religious practices.", "Rituals provide psychological comfort and social cohesion. Dismissing them ignores their documented benefits."),
    ("Your community is always causing trouble.", "Most religious communities run charities, schools, and hospitals. Their net contribution to society is overwhelmingly positive."),
    ("People from your religion don't belong here.", "A society that excludes people based on faith is neither free nor safe for anyone. Inclusion protects us all."),
    ("I can't believe anyone still follows that religion.", "Faith persists across millennia because it fulfills deep human needs for meaning, hope, and moral community."),
    ("Your faith makes you dangerous.", "Most faiths explicitly teach peace and compassion. Danger comes from extremism, not mainstream belief."),
    ("Everyone from your religion thinks the same way.", "Religious communities are incredibly diverse in theology, practice, and politics. Monolithic views are stereotypes."),
    ("Your religious practices are stupid.", "Practices that bring billions peace, community, and moral clarity are wise adaptations to human needs."),
    ("No one wants your religion around.", "Religious diversity enriches culture and fosters mutual understanding. Exclusion breeds ignorance and conflict."),
]

print(f"Base pairs defined: {len(BASE_PAIRS)}")

# ── Generate variations ─────────────────────────────────────────────

def generate_variations(pair, num=3):
    """Generate minor variations of a pair to expand dataset."""
    text, reply = pair
    results = [(text, reply)]
    
    # Variation strategies for aggressor text
    text_prefixes = [
        "", "Honestly, ", "Seriously, ", "I'm sick of this. ", ""
    ]
    text_suffixes = [
        "", " It's pathetic.", " Disgusting.", ""
    ]
    
    # Variation strategies for reply
    reply_prefixes = [
        "", "Please reconsider. ", "That's not okay. ", "Hold on. ", ""
    ]
    reply_suffixes = [
        "", " Let's keep discussions respectful.", " We can disagree without being disrespectful.", ""
    ]
    
    for _ in range(num - 1):
        new_text = random.choice(text_prefixes) + text + random.choice(text_suffixes)
        new_reply = random.choice(reply_prefixes) + reply + random.choice(reply_suffixes)
        results.append((new_text, new_reply))
    
    return results

# Expand base pairs
all_pairs = []
for pair in BASE_PAIRS:
    variations = generate_variations(pair, num=random.randint(2, 4))
    all_pairs.extend(variations)

# If still short of 1000, duplicate with random selection
while len(all_pairs) < 1000:
    pair = random.choice(BASE_PAIRS)
    all_pairs.append(pair)

# Shuffle and trim to exactly 1000
random.shuffle(all_pairs)
all_pairs = all_pairs[:1000]

# Categorize by class
categorized = {"ethnicity/race": [], "gender/sexual": [], "religion": []}
for text, reply in all_pairs:
    t_lower = text.lower()
    if any(w in t_lower for w in ["race", "ethnicity", "accent", "country", "heritage", "culture", "your people"]):
        categorized["ethnicity/race"].append((text, reply))
    elif any(w in t_lower for w in ["gender", "women", "men", "feminine", "sexual", "gay", "lgbt", "trans", "boy", "girl", "man ", "woman ", "marry", "family"]):
        categorized["gender/sexual"].append((text, reply))
    elif any(w in t_lower for w in ["religion", "faith", "pray", "holy", "god", "belief", "religious", "worship", "cult", "church"]):
        categorized["religion"].append((text, reply))
    else:
        categorized["gender/sexual"].append((text, reply))  # default

# Balance to ~333 per class
final_pairs = []
target_per_class = 334
for label, pairs in categorized.items():
    selected = pairs[:target_per_class]
    while len(selected) < target_per_class:
        selected.append(random.choice(pairs))
    for text, reply in selected:
        final_pairs.append({
            "cyberbullying_text": text,
            "de_escalatory_reply": reply,
            "label": label,
            "source_type": "synthetic_v2",
            "generation_strategy": "human_curated_de_escalation",
            "created_at": datetime.now().isoformat(),
        })

random.shuffle(final_pairs)

print(f"Final dataset: {len(final_pairs)} pairs")
print("Class distribution:")
for label in ["ethnicity/race", "gender/sexual", "religion"]:
    count = sum(1 for p in final_pairs if p["label"] == label)
    print(f"  {label}: {count}")

# Save CSV
csv_path = os.path.join("data", "synthetic", "reply_synthetic_v2.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["cyberbullying_text", "de_escalatory_reply", "label", "source_type", "generation_strategy", "created_at"])
    writer.writeheader()
    writer.writerows(final_pairs)
print(f"\n[Saved] CSV -> {csv_path}")

# Save JSON
json_path = os.path.join("data", "synthetic", "reply_synthetic_v2.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "total_pairs": len(final_pairs),
            "class_distribution": {label: sum(1 for p in final_pairs if p["label"] == label) for label in ["ethnicity/race", "gender/sexual", "religion"]},
            "generation_date": datetime.now().isoformat(),
            "source": "human-curated de-escalatory replies directed at aggressor",
            "description": "Cyberbullying text paired with calm, firm de-escalation aimed at the person saying the hate speech"
        },
        "pairs": final_pairs
    }, f, indent=2, ensure_ascii=False)
print(f"[Saved] JSON -> {json_path}")
