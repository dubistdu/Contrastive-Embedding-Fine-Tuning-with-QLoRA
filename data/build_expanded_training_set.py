"""
Build an expanded training set that mixes:
- 70% original synthetic pairs (structured prompts)
- 20% natural paraphrase-compatible pairs (from OOD eval)
- 10% hard negative incompatible pairs (hand-written, topic-aligned conflicts)

Why this helps:
- Mixing synthetic + natural phrasing reduces overfitting to the generator’s style
- Hard negatives teach the model that “same topic” ≠ “same preference”, improving OOD robustness

Usage:
  python data/build_expanded_training_set.py
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_minimal_jsonl(path: Path):
    """Load JSONL and normalize to {text_1, text_2, label}."""
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t1 = obj.get("text_1") or obj.get("text_a")
            t2 = obj.get("text_2") or obj.get("text_b")
            label = int(obj["label"])
            pairs.append({"text_1": t1, "text_2": t2, "label": label})
    return pairs


def write_jsonl(path: Path, pairs):
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def build_hard_negatives():
    """
    Hand-written hard negatives.

    These share topic and surface wording, but disagree on values or long-term
    goals. They avoid trivial mirroring like "I love X" vs "I hate X" and
    instead use softer, more realistic language.
    """
    examples = [
        ("I’d like to settle down in the next few years.", "I don’t really see myself settling in one place anytime soon."),
        ("I’m pretty sure I want kids at some point.", "I’ve never really pictured myself as a parent."),
        ("I’m trying to keep my weekends mostly quiet.", "Weekends are for going out and staying out late."),
        ("I’m hoping to stay close to my family.", "I’ve been actively putting distance between me and my family."),
        ("I save aggressively for the future.", "I’d rather enjoy my money now than worry about retirement."),
        ("I’m careful with substances.", "I like to experiment a bit when I’m out."),
        ("I’d rather talk through conflict early.", "I usually need days before I’m ready to talk about an issue."),
        ("I’m pretty rooted in this city.", "I move cities whenever I feel stuck."),
        ("I want a partner who loves hosting people.", "Having people over stresses me out."),
        ("I’m comfortable showing affection in public.", "Public displays of affection make me shut down."),
        ("I like to plan trips months ahead.", "Last-minute trips are the only ones that feel exciting."),
        ("I’d like a relationship that moves steadily forward.", "I’m wary of anything that starts to feel too serious."),
        ("I’m hoping to eventually buy a home.", "I don’t want to own property; I’d rather stay flexible."),
        ("I like having a small, tight-knit circle.", "I tend to collect a lot of acquaintances."),
        ("I check in with my partner during the day.", "I find constant texting kind of overwhelming."),
        ("I’m working toward a more minimalist lifestyle.", "I enjoy collecting things and having lots of stuff around."),
        ("I’m open to therapy and self-work.", "I don’t really believe talking about feelings changes much."),
        ("I’d like us to split household work fairly.", "I usually end up doing chores only when things are really bad."),
        ("I try to keep politics mostly off the table.", "Politics is a huge part of how I relate to people."),
        ("I like having a predictable routine.", "Doing the same thing two days in a row makes me restless."),
        ("I prefer to keep alcohol pretty minimal.", "I see drinks as a normal part of winding down most nights."),
        ("I’m trying to be more present and offline.", "I basically live on my phone—it’s how I stay connected."),
        ("I’m looking for one partner to build a life with.", "I’m curious about non-monogamy and don’t want to rule it out."),
        ("I want someone who shares my environmental values.", "I don’t think about environmental issues that much."),
        ("I enjoy cooking with a partner.", "Cooking feels like a chore I’d rather avoid."),
        ("I like to keep my home pretty orderly.", "I’m fine with piles and chaos as long as I can find things."),
        ("I’m careful with debt and big purchases.", "If I want something, I usually find a way to get it."),
        ("I’d rather spend free time outdoors.", "I prefer staying inside where it’s comfortable."),
        ("I’m trying to drink less these days.", "Most of my social life revolves around bars."),
        ("I’d like a partner who makes space for my creative work.", "Hobbies are nice but they shouldn’t take up too much time."),
        ("I see pets as part of the family.", "I’m okay with pets as long as they stay out of the way."),
        ("I value punctuality and showing up on time.", "I run late a lot; schedules feel flexible to me."),
        ("I’m intentional about checking in emotionally.", "I’d rather show I care through actions than long talks."),
        ("I’m hoping we’d both be open to compromise.", "I usually hold my ground once I’ve decided something."),
        ("I treat commitments as promises.", "Plans are more like suggestions to me."),
        ("I like to plan finances together.", "I prefer to keep money completely separate."),
        ("I recharge by spending quiet time at home.", "I only feel alive when I’m out with people."),
        ("I’m trying to cut back on screen time before bed.", "Scrolling is how I fall asleep most nights."),
        ("I’m comfortable talking about boundaries directly.", "I get uncomfortable when conversations turn serious."),
        ("I’d like to grow roots in one community.", "I get an itch to move every couple of years."),
        ("I’d rather work through problems than walk away.", "When things get hard, I usually shut down and withdraw."),
        ("I need a partner who takes mental health seriously.", "I think people rely on therapy more than they need to."),
        ("I’d like to start a family sooner rather than later.", "I’m in no rush and might not want kids at all."),
        ("I’m happy in a quieter neighborhood.", "I need to be in the middle of the action."),
        ("I want to prioritize our relationship over work.", "My career comes first and probably always will."),
        ("I’d rather be transparent about past relationships.", "I don’t really think the past needs to be talked about."),
        ("I see compromise as a core part of partnership.", "I’m used to doing things my way."),
        ("I like holidays to be low-key and calm.", "Holidays are for big parties and late nights."),
    ]
    return [{"text_1": a, "text_2": b, "label": 0} for a, b in examples]


def main():
    random.seed(42)

    synthetic_path = DATA_DIR / "dating_pairs.jsonl"
    ood_path = DATA_DIR / "ood_eval_pairs.jsonl"
    out_path = DATA_DIR / "dating_pairs_expanded.jsonl"

    synthetic = load_minimal_jsonl(synthetic_path)
    ood = load_minimal_jsonl(ood_path)

    # Natural paraphrase-compatible pairs come from the OOD file.
    natural_compat = [p for p in ood if p.get("bucket") == "paraphrase_compatible" or p["label"] == 1]

    hard_negs = build_hard_negatives()

    # Target proportions for the final training set.
    total = len(synthetic) + len(natural_compat) + len(hard_negs)
    n_syn = int(total * 0.7)
    n_nat = int(total * 0.2)
    n_hard = total - n_syn - n_nat

    # Sample from each pool (clamped if a pool is smaller than requested).
    syn_sample = random.sample(synthetic, k=min(n_syn, len(synthetic)))
    nat_sample = random.sample(natural_compat, k=min(n_nat, len(natural_compat)))
    hard_sample = random.sample(hard_negs, k=min(n_hard, len(hard_negs)))

    expanded = syn_sample + nat_sample + hard_sample
    random.shuffle(expanded)

    write_jsonl(out_path, expanded)

    print(f"Loaded {len(synthetic)} synthetic pairs from {synthetic_path}")
    print(f"Loaded {len(natural_compat)} natural paraphrase pairs from {ood_path}")
    print(f"Added {len(hard_sample)} hard negatives (topic-aligned conflicts).")
    print(f"Expanded training set size: {len(expanded)} → {out_path}")


if __name__ == "__main__":
    main()

