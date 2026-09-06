"""
COLA: Collaborative rOle-infused LLM-based Agents for Stance Detection
=======================================================================
Adapted from: "Stance Detection with Collaborative Role-Infused LLM-Based Agents"
              Lan et al., ICWSM 2024. https://arxiv.org/abs/2310.10467
Usage
-----
    # Set your key once:
    export OPENROUTER_API_KEY="sk-or-..."

    python cola.py --dataset trump
    python cola.py --dataset abortion --model meta-llama/llama-3.1-8b-instruct
    python cola.py                          
    python cola.py --data_dir split_datasets_enriched --samples 200
"""

import os
import json
import time
import argparse
import random
import numpy as np
from openai import OpenAI
from sklearn.metrics import f1_score
from collections import Counter

# Import shared utilities from llm_experiment to guarantee identical sampling
from llm_train import (
    get_openrouter_client,
    stratified_sample_users,
    load_data,
    call_api,
)


# ── COLA-specific helpers ─────────────────────────────────────────────────────

def call_llm(client, model: str, system: str, user: str,
             max_tokens: int = 300, retries: int = 3) -> str:
    """
    Thin wrapper around OpenRouter for COLA's multi-agent calls.
    Each agent has its own system prompt, so we can't use call_api directly.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"      API error ({e}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                print(f"      API error after {retries} attempts: {e}")
                return ""
    return ""


# ── user content builder ──────────────────────────────────────────────────────

def build_user_text(user_data: dict,
                    max_posts: int = 3,
                    max_comments: int = 5,
                    max_convos: int = 4,
                    max_chars: int = 2000) -> str:
    """
    Build compact text from user_data dict (same format as llm_experiment.py).
    Keys: posts, comments, conversations (list of {post_context, parent_content, user_reply}).
    """
    parts = []

    posts = user_data.get("posts", [])[:max_posts]
    if posts:
        parts.append("USER'S POSTS:")
        for p in posts:
            parts.append(f"  - {p[:400]}")

    comments = user_data.get("comments", [])[:max_comments]
    if comments:
        parts.append("\nUSER'S COMMENTS:")
        for c in comments:
            parts.append(f"  - {c[:300]}")

    convos = user_data.get("conversations", [])[:max_convos]
    if convos:
        parts.append("\nCONVERSATION EXCHANGES:")
        for convo in convos:
            post_ctx = convo.get("post_context", "")
            parent   = convo.get("parent_content", "")
            reply    = convo.get("user_reply", "")
            if post_ctx:
                parts.append(f"  [Under post]: {post_ctx[:200]}")
            if parent:
                parts.append(f"  [Original]: {parent[:200]}")
            parts.append(f"  [User reply]: {reply[:200]}")

    text = "\n".join(parts)
    return text[:max_chars]


TOPIC_DESCRIPTIONS = {
    "trump":      "Donald Trump and his political positions",
    "abortion":   "abortion rights and reproductive ethics",
    "religion":   "religion, faith, and philosophical worldviews",
}

# ── COLA Stage 1: Multidimensional Text Analysis ──────────────────────────────

ROLE_DISPLAY_NAMES = {
    "linguistic_expert"   : "Linguistic Expert",
    "domain_expert"       : "Domain Expert",
    "social_media_veteran": "Social Media Veteran",
}

STAGE1_ROLES = {
    "linguistic_expert": {
        "system": (
            "You are a Linguistic Expert specialising in stance detection on social media. "
            "Your task is to analyse the user's language from a purely linguistic standpoint."
        ),
        "focus": (
            "Analyse the following social media content and identify stance indicators from a "
            "LINGUISTIC perspective. Focus on:\n"
            "- Grammatical structure and sentence construction\n"
            "- Tense, modality, and hedging language\n"
            "- Rhetorical devices (irony, sarcasm, hyperbole, understatement)\n"
            "- Sentiment-bearing words and phrases\n"
            "- Discourse markers and argumentative connectors\n\n"
            "Be concise (max 150 words). Do NOT predict the stance yet."
        ),
    },
    "social_media_veteran": {
        "system": (
            "You are a Social Media Veteran who deeply understands online discourse culture. "
            "Your task is to analyse the user's content from a social media behaviour perspective."
        ),
        "focus": (
            "Analyse the following social media content and identify stance indicators from a "
            "SOCIAL MEDIA CULTURE perspective. Focus on:\n"
            "- Platform-specific expressions, slang, and abbreviations\n"
            "- Use of irony, memes, and satirical framing\n"
            "- In-group and out-group signalling language\n"
            "- Engagement patterns (who they respond to, how they respond)\n"
            "- Emotional tone and community identity markers\n\n"
            "Be concise (max 150 words). Do NOT predict the stance yet."
        ),
    },
}

# Dataset-specific domain expert replacing the generic "Domain Specialist"
DOMAIN_EXPERTS = {
    "trump": {
        "system": (
            "You are a Political Scientist specialising in American politics, populism, and "
            "electoral behaviour. You have deep expertise in Trump-era political discourse, "
            "Republican and Democrat ideological divides, and online political mobilisation."
        ),
        "focus": (
            "Analyse the following social media content and identify stance indicators from a "
            "POLITICAL SCIENCE perspective. Focus on:\n"
            "- Alignment with MAGA talking points, Republican or Democrat party positions\n"
            "- References to Trump policies, rallies, legal cases, or political figures\n"
            "- Use of political dog-whistles, wedge issues, and partisan framing\n"
            "- Expressions of political identity, loyalty, or opposition\n"
            "- Coded language used by Trump supporters or opponents online\n\n"
            "Be concise (max 150 words). Do NOT predict the stance yet."
        ),
    },
    "abortion": {
        "system": (
            "You are a Bioethicist and Reproductive Rights Advocate with expertise in the "
            "moral, legal, and medical dimensions of abortion discourse. You are deeply "
            "familiar with both pro-choice and pro-life philosophical frameworks."
        ),
        "focus": (
            "Analyse the following social media content and identify stance indicators from a "
            "BIOETHICS AND REPRODUCTIVE RIGHTS perspective. Focus on:\n"
            "- Moral and philosophical framing (bodily autonomy, personhood, sanctity of life)\n"
            "- References to legislation such as Roe v. Wade, Dobbs, or state-level laws\n"
            "- Medical or scientific claims about fetal development or maternal health\n"
            "- Religious or secular ethical grounding of arguments\n"
            "- Alignment with pro-choice, pro-life, or nuanced mixed-view positions\n\n"
            "Be concise (max 150 words). Do NOT predict the stance yet."
        ),
    },
    "religion": {
        "system": (
            "You are a Theologian and Comparative Religion Scholar with expertise across "
            "major world religions, atheism, agnosticism, and secular philosophy. You understand "
            "how religious identity is expressed, debated, and contested in online spaces."
        ),
        "focus": (
            "Analyse the following social media content and identify stance indicators from a "
            "THEOLOGY AND COMPARATIVE RELIGION perspective. Focus on:\n"
            "- Specific religious terminology, scripture references, or theological concepts\n"
            "- Expressions of faith, doubt, atheism, or agnosticism\n"
            "- Alignment with particular religious traditions (Christian, Islamic, Jewish, etc.)\n"
            "- Philosophical positions on the existence of God or the nature of belief\n"
            "- Attitudes toward organised religion, religious institutions, or secular society\n\n"
            "Be concise (max 150 words). Do NOT predict the stance yet."
        ),
    },
}


def stage1_analysis(client, model, topic, user_text, stance_classes):
    """
    Run the three role-infused agents and collect their analyses.
    Agent 1: Linguistic Expert (fixed)
    Agent 2: Dataset-specific Domain Expert (Political Scientist / Bioethicist / etc.)
    Agent 3: Social Media Veteran (fixed)
    Returns a dict: {role_name: analysis_text}
    """
    topic_desc  = TOPIC_DESCRIPTIONS.get(topic, topic)
    classes_str = ", ".join(f'"{c}"' for c in stance_classes)

    # Build the three agents: linguistic + domain-specific + social media
    domain_expert = DOMAIN_EXPERTS.get(topic, {
        "system": "You are a Domain Expert analysing social media content.",
        "focus":  (
            "Analyse the content from a domain knowledge perspective, focusing on "
            "topic-specific terminology, ideological alignment, and value framings. "
            "Be concise (max 150 words). Do NOT predict the stance yet."
        ),
    })

    agents = {
        "linguistic_expert" : STAGE1_ROLES["linguistic_expert"],
        "domain_expert"     : domain_expert,
        "social_media_veteran": STAGE1_ROLES["social_media_veteran"],
    }

    analyses = {}
    for role_name, role_cfg in agents.items():
        user_prompt = (
            f"Topic: {topic_desc}\n"
            f"Possible stances: {classes_str}\n\n"
            f"{role_cfg['focus']}\n\n"
            f"--- User Content ---\n{user_text}\n--- End ---"
        )
        analysis = call_llm(client, model,
                            system=role_cfg["system"],
                            user=user_prompt,
                            max_tokens=300)
        analyses[role_name] = {
            "system_prompt": role_cfg["system"],
            "user_prompt":   user_prompt,
            "response":      analysis,
        }

    return analyses


# ── COLA Stage 2: Reasoning-Enhanced Debating ─────────────────────────────────

def stage2_debate(client, model, topic, user_text,
                  stance_classes, analyses):
    """
    For each candidate stance, an Advocate argues why the user belongs there,
    drawing on the Stage 1 analyses.
    Returns a dict: {stance_class: advocate_argument}
    """
    topic_desc  = TOPIC_DESCRIPTIONS.get(topic, topic)
    # Extract just the response text from stage1 analyses for building prompts
    analyses_str = "\n\n".join(
        f"[{ROLE_DISPLAY_NAMES.get(role, role.replace('_', ' ').title())}]\n"
        f"{info['response'] if isinstance(info, dict) else info}"
        for role, info in analyses.items()
    )

    advocate_system = (
        "You are a Stance Advocate. Your job is to argue, based on provided analyses, "
        "why a social media user holds a particular stance. Be persuasive but grounded in evidence."
    )

    arguments = {}
    for stance in stance_classes:
        user_prompt = (
            f"Topic: {topic_desc}\n\n"
            f"Expert Analyses of the user's content:\n{analyses_str}\n\n"
            f"--- User Content ---\n{user_text}\n--- End ---\n\n"
            f"Argue specifically and concisely (max 100 words) why this user's stance is: \"{stance}\". "
            f"Draw evidence from the analyses above."
        )
        argument = call_llm(client, model,
                            system=advocate_system,
                            user=user_prompt,
                            max_tokens=200)
        arguments[stance] = {
            "system_prompt": advocate_system,
            "user_prompt":   user_prompt,
            "response":      argument,
        }

    return arguments


# ── COLA Stage 3: Stance Conclusion ───────────────────────────────────────────

def stage3_conclusion(client, model, topic, user_text,
                       stance_classes, analyses, arguments):
    """
    The Decision-Maker reads all advocate arguments and selects the final stance.
    Returns the predicted stance string.
    """
    topic_desc   = TOPIC_DESCRIPTIONS.get(topic, topic)
    classes_str  = ", ".join(f'"{c}"' for c in stance_classes)
    analyses_str = "\n\n".join(
        f"[{ROLE_DISPLAY_NAMES.get(role, role.replace('_', ' ').title())}]\n"
        f"{info['response'] if isinstance(info, dict) else info}"
        for role, info in analyses.items()
    )
    arguments_str = "\n\n".join(
        f"[Advocate for \"{stance}\"]\n"
        f"{info['response'] if isinstance(info, dict) else info}"
        for stance, info in arguments.items()
    )

    decision_system = (
        "You are the final Decision-Maker in a stance detection system. "
        "You will receive expert analyses and advocate arguments for each possible stance. "
        "Your task is to weigh all evidence and output ONLY the final stance label — "
        "nothing else, no explanation."
    )

    user_prompt = (
        f"Topic: {topic_desc}\n\n"
        f"Expert Analyses:\n{analyses_str}\n\n"
        f"Advocate Arguments:\n{arguments_str}\n\n"
        f"--- User Content ---\n{user_text}\n--- End ---\n\n"
        f"Based on all the evidence above, select the single best stance from: {classes_str}.\n"
        f"Reply with ONLY the exact stance label, nothing else."
    )

    raw = call_llm(client, model,
                   system=decision_system,
                   user=user_prompt,
                   max_tokens=50)

    # ── answer extraction with fuzzy fallback ──
    raw_clean = raw.strip().strip('"').strip("'")

    if raw_clean in stance_classes:
        final = raw_clean
    else:
        raw_lower = raw_clean.lower()
        matched = next(
            (sc for sc in stance_classes if sc.lower() == raw_lower), None)
        if not matched:
            candidates = [sc for sc in stance_classes
                          if sc.lower() in raw_lower or raw_lower in sc.lower()]
            matched = max(candidates, key=len) if candidates else raw_clean
        final = matched

    return {
        "system_prompt": decision_system,
        "user_prompt":   user_prompt,
        "raw_response":  raw,
        "final_label":   final,
    }


# ── per-user COLA pipeline ────────────────────────────────────────────────────

def run_cola_for_user(client, model, topic, user_data,
                       stance_classes, verbose=False):
    """
    Run full 3-stage COLA pipeline for a single user.
    Returns dict with prediction + full trace of prompts/responses,
    or None if user has no content.
    """
    user_text = build_user_text(user_data)

    if not user_text.strip():
        return None

    # Stage 1
    analyses = stage1_analysis(client, model, topic, user_text, stance_classes)

    # Stage 2
    arguments = stage2_debate(client, model, topic, user_text,
                               stance_classes, analyses)

    # Stage 3
    stage3 = stage3_conclusion(client, model, topic, user_text,
                               stance_classes, analyses, arguments)
    prediction = stage3["final_label"]

    if verbose:
        print(f"      Predicted: {prediction}")
        for role, info in analyses.items():
            print(f"\n  --- Stage 1 [{role}] ---")
            print(f"  PROMPT: {info['user_prompt'][:200]}...")
            print(f"  RESPONSE: {info['response'][:300]}...")
        for cls, info in arguments.items():
            print(f"\n  --- Stage 2 [{cls}] ---")
            print(f"  RESPONSE: {info['response'][:200]}...")
        print(f"\n  --- Stage 3 ---")
        print(f"  RAW: {stage3['raw_response']}  →  FINAL: {prediction}")

    return {
        "prediction": prediction,
        "user_text":  user_text,
        "stage1":     analyses,   # {role: {system_prompt, user_prompt, response}}
        "stage2":     arguments,  # {stance: {system_prompt, user_prompt, response}}
        "stage3":     stage3,     # {system_prompt, user_prompt, raw_response, final_label}
    }


# ── per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(client, model, dataset_name,
                data_dir="split_datasets_enriched_2",
                n_samples=200, seed=42, verbose=False):

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name.upper()}")
    print(f"  Model   : {model}")
    print(f"{'='*60}")

    # ── load data using the same function as llm_experiment.py ──
    data, stance_classes = load_data(dataset_name, data_dir)
    test_users_all = data["test"]

    # ── identical 200 users as llm_experiment.py (same seed + same function) ──
    test_users = stratified_sample_users(test_users_all, n_samples, seed)

    print(f"  Classes ({len(stance_classes)}): {stance_classes}")
    label_counts = Counter(n["label"] for n in test_users.values())
    print(f"  Sampled {len(test_users)} test users (stratified, seed={seed})")
    print(f"  Class distribution: {dict(label_counts)}")

    # ── run COLA pipeline ──
    y_true, y_pred = [], []
    skipped = 0
    full_traces = []   # store prompts + responses for each user

    for i, (uid, user_data) in enumerate(test_users.items()):
        true_label = user_data["label"]

        print(f"  [{i+1:>3}/{len(test_users)}] user={uid[:20]:<20}  "
              f"true={true_label:<25}", end="")

        result = run_cola_for_user(
            client, model,
            topic=dataset_name,
            user_data=user_data,
            stance_classes=stance_classes,
            verbose=verbose,
        )

        if result is None:
            print("  SKIPPED (no content)")
            skipped += 1
            continue

        pred = result["prediction"]

        # Map prediction to nearest valid class if not exact
        if pred not in stance_classes:
            pred_lower = pred.lower()
            match = next(
                (sc for sc in stance_classes if sc.lower() in pred_lower
                 or pred_lower in sc.lower()), stance_classes[0]
            )
            pred = match

        correct = "✓" if pred == true_label else "✗"
        print(f"  pred={pred:<25} {correct}")

        y_true.append(true_label)
        y_pred.append(pred)

        # Save full trace for this user
        full_traces.append({
            "user_id":    uid,
            "true_label": true_label,
            "pred_label": pred,
            "correct":    pred == true_label,
            "user_text":  result["user_text"],
            "stage1":     result["stage1"],
            "stage2":     result["stage2"],
            "stage3":     result["stage3"],
        })

        # Brief pause to respect rate limits
        time.sleep(0.3)

    # ── metrics ──
    if not y_true:
        print("  No valid predictions!")
        return None

    macro_f1 = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro",  zero_division=0)

    print(f"\n  ── Results ──")
    print(f"  Evaluated : {len(y_true)}  |  Skipped : {skipped}")
    print(f"  Macro F1  : {macro_f1*100:.2f}%")
    print(f"  Micro F1  : {micro_f1*100:.2f}%")

    return {
        "dataset"       : dataset_name,
        "model"         : model,
        "n_evaluated"   : len(y_true),
        "n_skipped"     : skipped,
        "num_classes"   : len(stance_classes),
        "macro_f1"      : round(macro_f1 * 100, 2),
        "micro_f1"      : round(micro_f1 * 100, 2),
        "predictions"   : [
            {"true": t, "pred": p} for t, p in zip(y_true, y_pred)
        ],
        "traces"        : full_traces,   # full prompt/response trace per user
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="COLA stance detection baseline for ControBench via OpenRouter")
    parser.add_argument("--dataset",  type=str, default=None,
                        help="Single dataset (default: all five)")
    parser.add_argument("--data_dir", type=str, default="split_datasets_enriched")
    parser.add_argument("--model",    type=str, default="openai/gpt-4o-mini",
                        help="OpenRouter model string (default: openai/gpt-4o-mini). "
                             "Other options: meta-llama/llama-3.1-8b-instruct, "
                             "google/gemini-flash-1.5")
    parser.add_argument("--samples",  type=int, default=200,
                        help="Users to sample per dataset (default: 200, "
                             "matching existing LLM baselines)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=str, default="cola_results.json")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print Stage 1-3 outputs for debugging")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Set your key: export OPENROUTER_API_KEY='sk-or-...'"
        )
    client = get_openrouter_client(api_key)

    all_datasets    = ["trump", "abortion", "religion"]
    datasets_to_run = [args.dataset] if args.dataset else all_datasets

    all_results = []
    for ds in datasets_to_run:
        result = run_dataset(
            client      = client,
            model       = args.model,
            dataset_name= ds,
            data_dir    = args.data_dir,
            n_samples   = args.samples,
            seed        = args.seed,
            verbose     = args.verbose,
        )
        if result:
            all_results.append(result)

    # ── summary table ──
    print("\n\n" + "="*68)
    print(f"  COLA  ({args.model})  --  Test Set Results")
    print("="*68)
    print(f"{'Dataset':<14} {'N':>5}  {'Macro F1':>10}  {'Micro F1':>10}")
    print("-"*68)
    for r in all_results:
        print(f"{r['dataset']:<14} {r['n_evaluated']:>5}  "
              f"{r['macro_f1']:>9.2f}%  {r['micro_f1']:>9.2f}%")
    print("="*68)

    # Save full results (including per-user predictions)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()