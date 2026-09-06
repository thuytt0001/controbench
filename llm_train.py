import json
import os
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import re


# ── OpenRouter client ─────────────────────────────────────────────────────────

def get_openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ── Deterministic stratified sampler ─────────────────────────────────────────

def stratified_sample_users(test_users: Dict, max_samples: int = 200,
                             seed: int = 42) -> Dict:
    """
    Sample exactly max_samples users with stratified class balance
    and a fixed seed — guarantees the same 200 users across all models.
    """
    random.seed(seed)
    np.random.seed(seed)

    by_class = {}
    for uid, udata in test_users.items():
        lbl = udata.get("label")
        if lbl:
            by_class.setdefault(lbl, []).append(uid)

    # Sort classes for determinism
    per_class = max(1, max_samples // len(by_class))
    sampled_ids = []

    for cls in sorted(by_class.keys()):
        ids = sorted(by_class[cls])          # sort for determinism
        random.shuffle(ids)                  # shuffle with fixed seed
        sampled_ids.extend(ids[:per_class])

    # Top up to max_samples if needed
    sampled_set = set(sampled_ids)
    remaining = sorted([uid for uid in test_users if uid not in sampled_set])
    random.shuffle(remaining)
    sampled_ids.extend(remaining[:max(0, max_samples - len(sampled_ids))])

    sampled_ids = sampled_ids[:max_samples]
    return {uid: test_users[uid] for uid in sampled_ids}


# ── Data loading (from test split) ────────────────────────────────────────────

def load_data(dataset: str,
              data_dir: str = "split_datasets_enriched_2_2",
              min_threshold: float = None) -> Tuple[Dict, List[str]]:
    """
    Load dataset from the pre-split test.json file.
    Label filtering matches split_dataset.py thresholds.
    """
    # ── load train for label vocabulary ──
    train_path = os.path.join(data_dir, dataset, "train.json")
    test_path  = os.path.join(data_dir, dataset, "test.json")

    print(f"📖 Loading {dataset} from test split: {test_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # ── derive valid labels from train (same as split_dataset.py) ──
    if min_threshold is None:
        min_threshold = 0.05 if dataset.lower() == "trump" else 0.01

    train_users_all = [n for n in train_data["nodes"] if n["type"] == "user"]
    total_train = len(train_users_all)
    label_counts = Counter(n["label"] for n in train_users_all if "label" in n)

    valid_labels = sorted([
        lbl for lbl, cnt in label_counts.items()
        if cnt / total_train >= min_threshold
    ])

    ignored = [(lbl, cnt) for lbl, cnt in label_counts.items()
               if cnt / total_train < min_threshold]
    if ignored:
        print(f"  Ignoring {len(ignored)} rare labels: "
              f"{[l for l,_ in ignored]}")

    print(f"  Valid labels ({len(valid_labels)}): {valid_labels}")

    # ── process test split ──
    test_users, _ = process_split(test_data, valid_labels)

    print(f"  Test users with valid labels: {len(test_users)}")

    return {"test": test_users}, valid_labels


def process_split(data: dict, valid_labels: List[str]) -> Tuple[Dict, Dict]:
    """Build user content dicts from a split JSON."""
    users = {}
    posts = {}

    for node in data["nodes"]:
        if node["type"] == "user":
            if node.get("label") in valid_labels:
                users[node["id"]] = {
                    "label":         node["label"],
                    "posts":         [],
                    "comments":      [],
                    "conversations": [],
                }
        elif node["type"] == "post":
            posts[node["id"]] = {
                "title":   str(node.get("title",   "")).strip(),
                "content": str(node.get("content", "")).strip(),
            }

    # ── index which posts each user commented on ──────────────────────────────
    # Used to find the post context for UCU conversations:
    # User A commented on Post X, User B replied to User A
    # → Post X is the conversation's parent post
    user_to_posts: Dict[str, List[str]] = {}  # user_id → [post_id, ...]
    for edge in data["edges"]:
        try:
            src, dst, etype = edge["source"], edge["target"], edge["type"]
            if etype in ("user_comment_post", "user_publish_post"):
                user_to_posts.setdefault(src, []).append(dst)
        except Exception:
            continue

    for edge in data["edges"]:
        try:
            src, dst, etype = edge["source"], edge["target"], edge["type"]

            if etype == "user_publish_post":
                if src in users and dst in posts:
                    t = posts[dst]["title"]
                    c = posts[dst]["content"]
                    if t and c:
                        text = f"Title: {t}\nContent: {c}" if t.lower() not in c.lower() else c
                    elif t:
                        text = f"Title: {t}"
                    elif c:
                        text = c
                    else:
                        continue
                    users[src]["posts"].append(text)

            elif etype == "user_comment_post":
                if src in users and dst in posts:
                    content = str(edge.get("content", "")).strip()
                    if len(content) > 5:
                        t = posts[dst]["title"]
                        c = posts[dst]["content"]
                        if t and c:
                            ctx = f"Title: {t}\nContent: {c[:200]}..."
                        elif t:
                            ctx = f"Title: {t}"
                        elif c:
                            ctx = c[:200] + "..."
                        else:
                            ctx = "[Empty post]"
                        users[src]["comments"].append(
                            f"Post context: ({ctx}) → User's comment: {content}"
                        )

            elif etype == "user_comment_user":
                # src = User A (original commenter)
                # dst = User B (replier) — THIS is who we predict
                if dst in users:
                    user_a_original = str(edge.get("content",      "")).strip()
                    user_b_reply    = str(edge.get("reply_content", "")).strip()
                    # LEAKAGE FIX: never expose the neighbor's ground-truth stance
                    # label. The model must infer the target user's stance from
                    # content alone, not from who they talked to.
                    target_label    = "another"

                    # Find the post this conversation happened under:
                    # User A's post edges give us the candidate posts
                    post_ctx = ""
                    src_post_ids = user_to_posts.get(src, [])
                    for pid in src_post_ids:
                        if pid in posts:
                            t = posts[pid]["title"]
                            c = posts[pid]["content"]
                            if t and c:
                                post_ctx = f"Title: {t}\nContent: {c[:300]}..."
                            elif t:
                                post_ctx = f"Title: {t}"
                            elif c:
                                post_ctx = c[:300] + "..."
                            break  # use first matching post

                    users[dst]["conversations"].append({
                        "target_user":    src,
                        "target_opinion": target_label,
                        "user_reply":     user_b_reply,
                        "parent_content": user_a_original,
                        "post_context":   post_ctx,
                    })

        except Exception as e:
            continue

    return users, posts


# ── Prompt builder ────────────────────────────────────────────────────────────

def create_prompt(dataset: str, posts: List[str], comments: List[str],
                  conversations: List[Dict], categories: List[str]) -> str:

    # Posts
    posts_text = "No posts available"
    if posts:
        posts_text = ""
        for i, post in enumerate(posts[:4]):
            short = post[:2000] + "..." if len(post) > 2000 else post
            posts_text += f"\nPost {i+1}: {short}\n" + "-"*40

    # Comments
    comments_text = "No comments available"
    if comments:
        comments_text = ""
        for i, comment in enumerate(comments[:6]):
            short = comment[:3000] + "..." if len(comment) > 3000 else comment
            comments_text += f"\nComment {i+1}: {short}\n" + "-"*30

    # Conversations
    conversations_text = ""
    if conversations:
        conversations_text = "\nCONVERSATION EXCHANGES:\n"
        for i, conv in enumerate(conversations[:8]):
            conversations_text += f"\nConversation {i+1}:\n"
            post_ctx = conv.get("post_context", "")
            parent   = conv.get("parent_content", "")
            reply    = conv.get("user_reply", "")
            # LEAKAGE FIX: neighbor stance label is no longer read or shown.
            if post_ctx:
                conversations_text += f"  Under post: ({post_ctx})\n"
            if parent:
                short_p = parent[:800] + "..." if len(parent) > 800 else parent
                conversations_text += f"  Original comment by another user:\n"
                conversations_text += f"  → \"{short_p}\"\n\n"
            if reply:
                short_r = reply[:1000] + "..." if len(reply) > 1000 else reply
                conversations_text += f"  User's reply:\n"
                conversations_text += f"  → \"{short_r}\"\n"
            conversations_text += "-"*50 + "\n"

        # LEAKAGE FIX: the INTERACTION SUMMARY previously listed per-stance
        # neighbor counts (e.g. "5 conversations with Pro-Choice users"),
        # which directly leaks neighbor ground-truth labels. It has been
        # replaced with a label-free total count.
        conversations_text += (
            f"\nINTERACTION SUMMARY: {len(conversations)} total "
            f"conversation exchanges.\n"
        )

    # Category hints
    HINTS = {
        "religion": {
            "Christian": " (includes Catholic, Protestant, Orthodox, etc.)",
            "Islamic":   " (includes Muslim, Sunni, Shia, etc.)",
            "philosophical/other": " (includes spiritual, deist, humanist, etc.)",
        },
        "abortion": {
            "Pro-Life":   " (opposes abortion, supports fetal rights)",
            "Pro-Choice": " (supports reproductive choice, women's rights)",
            "Mixed View": " (nuanced position, contextual support)",
        },
    }
    categories_text = ""
    for cat in categories:
        hint = HINTS.get(dataset, {}).get(cat, "")
        categories_text += f"• {cat}{hint}\n"

    prompt = f"""ACADEMIC RESEARCH ANALYSIS: This is an objective analysis of social media content for academic research on opinion classification. The goal is to categorize user perspectives based on their communication patterns, not to promote any particular viewpoint.
You are an expert at analyzing social media conversations to understand people's beliefs and opinions.
TASK: Analyze this user's posts, comments, and conversation exchanges to determine their stance on {dataset}.

USER'S POSTS:
{posts_text}

USER'S COMMENTS ON POSTS:
{comments_text}

{conversations_text}

ANALYSIS STEPS:
1. What beliefs do they express in their own posts?
2. What do their comments on posts reveal about their views?
3. How do they engage in conversations - what do they reply to and how?
4. What does the content of their replies reveal about their own position?
5. What patterns emerge from the full conversation context?
6. Which category best matches their overall worldview?

AVAILABLE CATEGORIES:
{categories_text}

Pay special attention to:
- The content of what they're replying to (shows what triggers their responses)
- How they respond to different viewpoints
- Whether they challenge, support, or provide nuanced takes
- The tone and approach in their conversation exchanges

Think step by step about the evidence, then select the single best category.

REASONING:
1. Key beliefs from posts:
2. Patterns from post comments:
3. Conversation engagement patterns:
4. Response style and tone:
5. Best category match:

IMPORTANT: End your response with exactly one line containing only your final answer.
FINAL ANSWER: [Select exactly one category from the available categories list above]"""

    return prompt


# ── Answer extractor ──────────────────────────────────────────────────────────

SEMANTIC_MAPPINGS = {
    "trump supporter":     ["trump supporter", "pro trump", "maga", "republican",
                            "conservative", "support trump", "trump"],
    "non-trump supporter": ["non-trump supporter", "anti trump", "democrat",
                            "liberal", "against trump", "oppose trump"],
    "christian":    ["christian", "catholic", "protestant", "orthodox", "evangelical", "baptist"],
    "islamic":      ["islamic", "muslim", "sunni", "shia", "islam"],
    "jewish":       ["jewish", "judaism", "jew"],
    "buddhist":     ["buddhist", "buddhism"],
    "hindu":        ["hindu", "hinduism"],
    "non-theistic": ["non-theistic", "atheist", "agnostic", "secular", "non-religious"],
    "philosophical/other": ["philosophical", "spiritual", "deist", "humanist"],
    "pro-life":     ["pro-life", "pro life", "right to life", "anti abortion", "prolife"],
    "pro-choice":   ["pro-choice", "pro choice", "reproductive rights", "prochoice"],
    "mixed view":   ["mixed view", "mixed", "moderate", "nuanced"],
}


def extract_answer(response: str, categories: List[str]) -> Tuple[str, str]:
    if response == "ERROR":
        return categories[0], "api_error"

    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if not lines:
        return categories[0], "no_match"

    # 1. Explicit FINAL ANSWER: pattern
    answer = None
    for line in reversed(lines):
        m = re.search(r"FINAL ANSWER:\s*(.+)", line, re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
            break

    # 2. Scan last 5 lines for category match
    if not answer:
        for line in reversed(lines[-5:]):
            clean = re.sub(r'^(the user is|answer:|category:|based on|'
                           r'this suggests|while|however)\s*', '',
                           line.strip('"\'•-*[]()'), flags=re.IGNORECASE).strip()
            for cat in categories:
                if cat.lower() in clean.lower() or clean.lower() in cat.lower():
                    answer = clean
                    break
            if answer:
                break

    if not answer:
        answer = lines[-1].strip('"\'•-*[]()').strip()

    answer = re.sub(r'^(the user is|answer:|category:|based on|'
                    r'this suggests|while|however)\s*', '',
                    answer, flags=re.IGNORECASE).strip('"\'•-*[]()').strip()

    # Exact match
    for cat in categories:
        if cat.lower() == answer.lower():
            return cat, "exact"

    # Partial match
    for cat in categories:
        if answer.lower() in cat.lower() or cat.lower() in answer.lower():
            return cat, "partial"

    # Semantic match
    answer_lower = answer.lower()
    for canonical, variants in SEMANTIC_MAPPINGS.items():
        matching_cat = next((c for c in categories
                             if c.lower() == canonical), None)
        if matching_cat:
            for variant in variants:
                if variant in answer_lower:
                    return matching_cat, "semantic"

    return categories[0], "no_match"


# ── LLM caller (OpenRouter) ───────────────────────────────────────────────────

# Thinking models produce long chain-of-thought before answering —
# they need higher token limits, more retries, and longer waits
THINKING_MODELS = {
    "deepseek/deepseek-r1",
    "qwen/qwen3-235b-a22b:thinking",
    "qwen/qwen3-235b-a22b-thinking-2507",
}

def call_api(client: OpenAI, model: str, prompt: str,
             max_tokens: int = 500, retries: int = 3) -> Tuple[str, int]:

    is_thinking = any(m in model for m in THINKING_MODELS)
    if is_thinking:
        retries    = 5            # more retries for thinking models
        max_tokens = max(max_tokens, 8000)  # thinking needs more tokens

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": ("You are a neutral and precise analyst who analyses "
                                 "conversation patterns and content. Follow the analysis "
                                 "steps and select exactly one category.")},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            text   = response.choices[0].message.content.strip()
            tokens = getattr(response.usage, "total_tokens", 0)
            return text, tokens
        except Exception as e:
            if attempt < retries - 1:
                # Thinking models: longer exponential backoff (5s, 10s, 20s, 40s)
                # Standard models: short backoff (1s, 2s, 4s)
                base_wait = 5 if is_thinking else 2
                wait = base_wait ** (attempt + 1)
                print(f"      API error ({e}), retrying in {wait}s …", flush=True)
                time.sleep(wait)
            else:
                print(f"      API failed after {retries} attempts: {e}", flush=True)
                return "ERROR", 0
    return "ERROR", 0


# ── Predictor class ───────────────────────────────────────────────────────────

class OpinionPredictor:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model  = model
        self.total_tokens = 0

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Sanitise model name for folder (replace / with _)
        model_slug = model.replace("/", "_").replace(":", "_")
        self.experiment_folder = f"llm_experiment_{model_slug}_{timestamp}"
        os.makedirs(self.experiment_folder, exist_ok=True)

        self.results_file   = os.path.join(self.experiment_folder, "results.txt")
        self.responses_file = os.path.join(self.experiment_folder, "llm_responses.txt")
        self.summary_file   = os.path.join(self.experiment_folder, "experiment_summary.txt")
        self._init_files()

        print(f"🚀 Initialized {model}")
        print(f"📁 Experiment folder: {self.experiment_folder}")

    def _init_files(self):
        header = (f"Model: {self.model}\n"
                  f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"API: OpenRouter\n" + "="*80 + "\n\n")
        for path, title in [(self.results_file,   "EXPERIMENT RESULTS LOG"),
                            (self.responses_file, "LLM RESPONSES LOG"),
                            (self.summary_file,   "EXPERIMENT SUMMARY")]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{title}\n{'='*80}\n{header}")

    def log(self, path, msg):
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg)

    def predict_user(self, dataset, user_id, user_data, categories):
        prompt = create_prompt(
            dataset       = dataset,
            posts         = user_data["posts"],
            comments      = user_data["comments"],
            conversations = user_data["conversations"],
            categories    = categories,
        )
        self.log(self.responses_file,
                 f"\n{'='*80}\nPROMPT for {user_id}:\n{prompt}\n{'='*80}\n\n")

        response, tokens = call_api(self.client, self.model, prompt)
        self.total_tokens += tokens

        self.log(self.responses_file,
                 f"RESPONSE for {user_id}:\n{response}\n{'='*80}\n\n")

        prediction, match_type = extract_answer(response, categories)
        return prediction, match_type, tokens


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_experiment(dataset: str, client: OpenAI, model: str,
                   data_dir: str = "split_datasets_enriched_2",
                   max_samples: int = 200, seed: int = 42) -> Dict:

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset.upper()}")
    print(f"  Model   : {model}")
    print(f"{'='*60}")

    data, categories = load_data(dataset, data_dir)
    test_users_all   = data["test"]

    # ── deterministic stratified sample — same 200 users for every model ──
    test_users = stratified_sample_users(test_users_all, max_samples, seed)

    label_dist = Counter(u["label"] for u in test_users.values())
    print(f"  Sampled {len(test_users)} users (stratified, seed={seed})")
    print(f"  Class distribution: {dict(label_dist)}")

    predictor   = OpinionPredictor(client, model)
    predictions = []
    true_labels = []
    match_types = Counter()
    detailed    = []
    correct     = 0

    header = (f"\n{'#'*80}\nEXPERIMENT: {model} on {dataset.upper()}\n"
              f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"Categories: {categories}\n{'#'*80}\n\n")
    predictor.log(predictor.results_file,   header)
    predictor.log(predictor.responses_file, header)
    predictor.log(predictor.summary_file,   header)

    for user_id, user_data in tqdm(test_users.items(), desc=f"{dataset}"):
        pred, match_type, tokens = predictor.predict_user(
            dataset, user_id, user_data, categories)

        true = user_data["label"]
        is_correct = pred == true
        if is_correct:
            correct += 1

        predictions.append(pred)
        true_labels.append(true)
        match_types[match_type] += 1
        detailed.append({
            "user_id":        user_id,
            "true_label":     true,
            "prediction":     pred,
            "correct":        is_correct,
            "match_type":     match_type,
            "tokens":         tokens,
            "num_posts":      len(user_data["posts"]),
            "num_comments":   len(user_data["comments"]),
            "num_convos":     len(user_data["conversations"]),
        })

        status = "✅" if is_correct else "❌"
        predictor.log(predictor.results_file,
                      f"{status} {user_id} | true={true} | pred={pred} | "
                      f"match={match_type}\n")

        time.sleep(0.1)   # gentle rate-limit buffer

    # ── metrics ──
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro",  zero_division=0)
    micro_f1 = f1_score(true_labels, predictions, average="micro",  zero_division=0)

    summary = (
        f"\n{'#'*80}\nRESULTS: {model} on {dataset.upper()}\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Macro F1 : {macro_f1:.4f}\n"
        f"Micro F1 : {micro_f1:.4f}\n"
        f"Correct  : {correct}/{len(test_users)}\n"
        f"Tokens   : {predictor.total_tokens:,}\n"
        f"Match types: {dict(match_types)}\n{'#'*80}\n"
    )
    predictor.log(predictor.results_file, summary)
    predictor.log(predictor.summary_file, summary)
    print(summary)

    return {
        "dataset":           dataset,
        "model":             model,
        "accuracy":          accuracy,
        "macro_f1":          macro_f1,
        "micro_f1":          micro_f1,
        "predictions":       predictions,
        "true_labels":       true_labels,
        "match_types":       dict(match_types),
        "total_tokens":      predictor.total_tokens,
        "detailed_results":  detailed,
        "experiment_folder": predictor.experiment_folder,
    }


# ── Multi-dataset / multi-model runner ───────────────────────────────────────

def run_all_experiments(datasets: List[str], api_key: str,
                        models: List[str],
                        data_dir: str = "split_datasets_enriched_2",
                        max_samples: int = 200, seed: int = 42) -> Tuple[Dict, str]:

    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    main_folder = f"llm_experiments_{timestamp}"
    os.makedirs(main_folder, exist_ok=True)

    client      = get_openrouter_client(api_key)
    all_results = {}

    for dataset in datasets:
        all_results[dataset] = {}
        for model in models:
            result = run_experiment(dataset, client, model,
                                    data_dir, max_samples, seed)
            all_results[dataset][model] = result

    # ── overall summary ──
    summary_path = os.path.join(main_folder, "overall_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"OVERALL SUMMARY\n{'='*80}\n"
                f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Datasets : {datasets}\n"
                f"Models   : {models}\n"
                f"Samples  : {max_samples} stratified users per dataset (seed={seed})\n"
                f"API      : OpenRouter\n{'='*80}\n\n")
        f.write(f"{'Dataset':<14} {'Model':<35} {'Accuracy':<10} "
                f"{'Macro F1':<10} {'Micro F1':<10} {'Tokens':<10}\n")
        f.write("-"*90 + "\n")
        for ds, ds_results in all_results.items():
            for mdl, r in ds_results.items():
                f.write(f"{ds:<14} {mdl:<35} {r['accuracy']:<10.4f} "
                        f"{r['macro_f1']:<10.4f} {r['micro_f1']:<10.4f} "
                        f"{r['total_tokens']:<10,}\n")

    print(f"\n📁 All results in: {main_folder}")
    print(f"📋 Summary: {summary_path}")
    return all_results, main_folder


def save_and_analyze(results: Dict, main_folder: str):
    rows = []
    for dataset, ds_results in results.items():
        for model, r in ds_results.items():
            rows.append({
                "Dataset":  dataset,
                "Model":    model,
                "Accuracy": f"{r['accuracy']:.4f}",
                "Macro_F1": f"{r['macro_f1']:.4f}",
                "Micro_F1": f"{r['micro_f1']:.4f}",
                "Tokens":   f"{r['total_tokens']:,}",
                "Folder":   r["experiment_folder"],
            })

    df = pd.DataFrame(rows)
    print(f"\n{'='*60}\n🏆 RESULTS SUMMARY\n{'='*60}")
    print(df.to_string(index=False))

    csv_path = os.path.join(main_folder, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📊 CSV saved: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── configuration ──────────────────────────────────────────────────────
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "Set your key: export OPENROUTER_API_KEY='sk-or-...'"
        )

    MODELS = [
        "openai/gpt-4o-mini",                    # GPT-4o-mini
        "meta-llama/llama-3.1-8b-instruct",      # Llama-3.1-8B
        "deepseek/deepseek-chat-v3-0324",        # DeepSeek-V3
        "deepseek/deepseek-r1",                  # DeepSeek-R1 (reasoning)
        "qwen/qwen3-235b-a22b",                  # Qwen3-235B
        "qwen/qwen3-235b-a22b-thinking-2507",    # Qwen3-235B-Thinking
        "moonshotai/kimi-k2",                       # Kimi K2
    ]

    DATASETS = ["trump", "abortion", "religion"]

    DATA_DIR    = "split_datasets_enriched_2"
    MAX_SAMPLES = 200      # same 200 stratified users for every model
    SEED        = 42       # fixed seed — guarantees identical sample across models
    # ───────────────────────────────────────────────────────────────────────

    print("🚀 LLM OPINION PREDICTOR  (OpenRouter)")
    print(f"   Models   : {MODELS}")
    print(f"   Datasets : {DATASETS}")
    print(f"   Samples  : {MAX_SAMPLES} stratified users per dataset (seed={SEED})")
    print(f"   Data dir : {DATA_DIR}")
    print("="*70)

    results, main_folder = run_all_experiments(
        datasets    = DATASETS,
        api_key     = OPENROUTER_API_KEY,
        models      = MODELS,
        data_dir    = DATA_DIR,
        max_samples = MAX_SAMPLES,
        seed        = SEED,
    )

    save_and_analyze(results, main_folder)
    print("\n🎉 DONE!")