
"""
Text Classification Experiments with Train/Val/Test Split (60:20:20)
- Hyperparameter tuning on validation split ONLY.
- Final training on (train + val) and a single evaluation on test.
"""

import json
import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import warnings
import itertools
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Optional libraries
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
SIMCSE_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except Exception as e:
    print(f"❌ PyTorch not available: {e}")

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    from transformers import (
        AutoTokenizer, AutoModel,
        BertTokenizer, BertForSequenceClassification,
        RobertaTokenizer, RobertaForSequenceClassification,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available")
except Exception as e:
    print(f"❌ Transformers not available: {e}")

try:
    from simcse import SimCSE
    SIMCSE_AVAILABLE = True
    print("✅ SimCSE available")
except Exception as e:
    print(f"❌ SimCSE not available: {e}")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ Sentence-Transformers (SBERT) available")
except Exception as e:
    print(f"❌ Sentence-Transformers not available: {e}")

import matplotlib.pyplot as plt
from datetime import datetime
import time


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


set_seed(42)


class SplitDatasetProcessor:
    """Process pre-split graph datasets for text classification."""

    def __init__(self, dataset_name: str, split_dir: str = "split_datasets_enriched_2"):
        self.dataset_name = dataset_name
        self.split_dir = split_dir

    def load_split_data(self) -> Tuple[Dict, Dict, Dict, List[str]]:
        """Load train, validation, and test data from pre-split files."""

        dataset_dir = os.path.join(self.split_dir, self.dataset_name)
        train_file = os.path.join(dataset_dir, "train.json")
        val_file = os.path.join(dataset_dir, "validation.json")
        test_file = os.path.join(dataset_dir, "test.json")

        print(f"📖 Loading {self.dataset_name} splits from: {dataset_dir}")

        for file_path in [train_file, val_file, test_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Split file not found: {file_path}")

        with open(train_file, "r", encoding='utf-8') as f:
            train_data = json.load(f)
        with open(val_file, "r", encoding='utf-8') as f:
            val_data = json.load(f)
        with open(test_file, "r", encoding='utf-8') as f:
            test_data = json.load(f)

        train_processed = self._process_split_data(train_data)
        val_processed = self._process_split_data(val_data)
        test_processed = self._process_split_data(test_data)

        all_labels = set()
        for split_data in [train_processed, val_processed, test_processed]:
            all_labels.update(user_data["label"] for user_data in split_data.values())
        labels = sorted(list(all_labels))

        print(f"✅ Loaded splits: {len(train_processed)} train, {len(val_processed)} val, {len(test_processed)} test")
        print(f"📊 Labels: {labels}")
        return train_processed, val_processed, test_processed, labels

    def _process_split_data(self, data: Dict) -> Dict:
        """Process a single split of the dataset into user-centric text fields."""
        users, posts = {}, {}

        for node in data["nodes"]:
            if node["type"] == "user":
                users[node["id"]] = {
                    "label": node["label"],
                    "posts": [],
                    "comments_on_posts": [],
                    "comments_in_conversations": [],
                    "replies_in_conversations": []
                }
            elif node["type"] == "post":
                posts[node["id"]] = {
                    "title": str(node.get("title", "")).strip(),
                    "content": str(node.get("content", "")).strip(),
                    "author": None
                }

        for edge in data["edges"]:
            try:
                et = edge["type"]
                if et == "user_publish_post":
                    u, p = edge["source"], edge["target"]
                    if u in users and p in posts:
                        posts[p]["author"] = u
                        text = self._combine_post_text(posts[p])
                        if text:
                            users[u]["posts"].append(text)

                elif et == "user_comment_post":
                    u = edge["source"]
                    if u in users and "content" in edge:
                        t = str(edge["content"]).strip()
                        if len(t) > 10:
                            users[u]["comments_on_posts"].append(t)

                elif et == "user_comment_user":
                    ua, ub = edge["source"], edge["target"]
                    if ua in users and ub in users:
                        if "content" in edge:
                            oc = str(edge["content"]).strip()
                            if len(oc) > 10:
                                users[ua]["comments_in_conversations"].append(oc)
                        if "reply_content" in edge:
                            rc = str(edge["reply_content"]).strip()
                            if len(rc) > 10:
                                users[ub]["replies_in_conversations"].append(rc)
            except Exception as e:
                print(f"Warning: Error processing edge: {e}")
                continue

        return users

    def _combine_post_text(self, post: Dict) -> str:
        title = post["title"]
        content = post["content"]
        if title and content:
            return content if title.lower() in content.lower() else f"{title} {content}"
        return title or content or ""

    def create_text_features(self, users: Dict) -> Tuple[List[str], List[str], List[Dict]]:
        texts, labels, metadata = [], [], []
        for uid, ud in users.items():
            all_texts = ud["posts"] + ud["comments_on_posts"] + ud["comments_in_conversations"] + ud["replies_in_conversations"]
            if all_texts:
                combined = " [SEP] ".join(all_texts)
                texts.append(combined)
                labels.append(ud["label"])
                metadata.append({
                    "user_id": uid,
                    "num_posts": len(ud["posts"]),
                    "num_comments": len(ud["comments_on_posts"]),
                    "num_conversations": len(ud["comments_in_conversations"]) + len(ud["replies_in_conversations"]),
                    "total_text_length": len(combined)
                })
        return texts, labels, metadata


class CustomDataset(Dataset):
    """Custom dataset for transformer models (single-label classification)."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        unique_labels = sorted(list(set(labels)))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label2id[label], dtype=torch.long)
        }


class TextClassificationExperiments:
    """Run comprehensive text classification experiments with hyperparameter tuning."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"text_experiments_{dataset_name}_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.results = {}
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")

        self.results_file = os.path.join(self.experiment_dir, "results.txt")
        self.hyperparams_file = os.path.join(self.experiment_dir, "hyperparameter_search.txt")

        with open(self.results_file, "w") as f:
            f.write(f"Text Classification Experiments - {dataset_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Using Train/Validation/Test Split (60:20:20)\n")
            f.write(f"PyTorch: {TORCH_AVAILABLE}, Transformers: {TRANSFORMERS_AVAILABLE}\n")
            f.write(f"SimCSE: {SIMCSE_AVAILABLE}, SBERT: {SENTENCE_TRANSFORMERS_AVAILABLE}\n")
            f.write("="*80 + "\n\n")

        with open(self.hyperparams_file, "w") as f:
            f.write(f"Hyperparameter Search Results - {dataset_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

    def log_result(self, message: str):
        print(message)
        with open(self.results_file, "a") as f:
            f.write(message + "\n")

    def log_hyperparams(self, message: str):
        print(message)
        with open(self.hyperparams_file, "a") as f:
            f.write(message + "\n")

    # ---------------- Baselines ----------------

    def run_baseline_experiments(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.log_result("\n" + "="*60)
        self.log_result("BASELINE EXPERIMENTS")
        self.log_result("="*60)

        # Random
        self.log_result("\n1. Random Classifier")
        random_clf = DummyClassifier(strategy="uniform", random_state=42)
        random_clf.fit(X_train, y_train)
        y_pred_random = random_clf.predict(X_test)
        self.results["Random"] = {
            "accuracy": accuracy_score(y_test, y_pred_random),
            "macro_f1": f1_score(y_test, y_pred_random, average='macro', zero_division=0),
            "micro_f1": f1_score(y_test, y_pred_random, average='micro', zero_division=0)
        }
        self.log_result(f"Accuracy: {self.results['Random']['accuracy']:.4f}")
        self.log_result(f"Macro F1: {self.results['Random']['macro_f1']:.4f}")
        self.log_result(f"Micro F1: {self.results['Random']['micro_f1']:.4f}")

        # Majority
        self.log_result("\n2. Majority Class Classifier")
        majority_clf = DummyClassifier(strategy="most_frequent", random_state=42)
        majority_clf.fit(X_train, y_train)
        y_pred_majority = majority_clf.predict(X_test)
        self.results["Majority"] = {
            "accuracy": accuracy_score(y_test, y_pred_majority),
            "macro_f1": f1_score(y_test, y_pred_majority, average='macro', zero_division=0),
            "micro_f1": f1_score(y_test, y_pred_majority, average='micro', zero_division=0)
        }
        self.log_result(f"Accuracy: {self.results['Majority']['accuracy']:.4f}")
        self.log_result(f"Macro F1: {self.results['Majority']['macro_f1']:.4f}")
        self.log_result(f"Micro F1: {self.results['Majority']['micro_f1']:.4f}")

    # ---------------- TF-IDF + LogReg ----------------

    def tune_tfidf_logistic_regression(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.log_hyperparams("\n🔧 TF-IDF + Logistic Regression Hyperparameter Tuning")
        self.log_hyperparams("-" * 55)

        param_combinations = [
            (10000, (1, 2), 1.0, 'lbfgs'),
            (5000, (1, 1), 0.1, 'liblinear'),
            (20000, (1, 3), 10.0, 'saga'),
            (10000, (1, 1), 1.0, 'liblinear'),
            (10000, (1, 2), 0.1, 'saga'),
            (5000, (1, 2), 1.0, 'lbfgs'),
            (20000, (1, 2), 1.0, 'saga'),
            (10000, (1, 3), 1.0, 'saga'),
        ]

        best_score, best_params = 0, None
        self.log_hyperparams(f"Testing {len(param_combinations)} parameter combinations...")

        for i, (max_features, ngram_range, C, solver) in enumerate(param_combinations):
            try:
                vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=ngram_range)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_val_tfidf = vectorizer.transform(X_val)

                lr_clf = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', C=C, solver=solver)
                lr_clf.fit(X_train_tfidf, y_train)
                y_pred_val = lr_clf.predict(X_val_tfidf)

                val_f1_macro = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
                val_acc = accuracy_score(y_val, y_pred_val)

                self.log_hyperparams(f"Config {i+1}: features={max_features}, ngrams={ngram_range}, C={C}, solver={solver}")
                self.log_hyperparams(f"   Validation F1: {val_f1_macro:.4f}, Accuracy: {val_acc:.4f}")

                if val_f1_macro > best_score:
                    best_score = val_f1_macro
                    best_params = {'max_features': max_features, 'ngram_range': ngram_range, 'C': C, 'solver': solver}
            except Exception as e:
                self.log_hyperparams(f"Error with params {(max_features, ngram_range, C, solver)}: {e}")
                continue

        if best_params is None:
            self.log_result("\n❌ TF-IDF+LogReg hyperparameter tuning failed")
            self.results["TF-IDF+LogReg"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "Tuning failed"}
            return

        self.log_hyperparams("\n🏆 Best TF-IDF+LogReg parameters:")
        for k, v in best_params.items():
            self.log_hyperparams(f"   {k}: {v}")
        self.log_hyperparams(f"   Validation F1: {best_score:.4f}")

        vectorizer = TfidfVectorizer(max_features=best_params['max_features'], stop_words='english', ngram_range=best_params['ngram_range'])
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        final_clf = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', C=best_params['C'], solver=best_params['solver'])
        final_clf.fit(X_train_tfidf, y_train)
        y_pred_test = final_clf.predict(X_test_tfidf)

        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        test_f1_micro = f1_score(y_test, y_pred_test, average='micro', zero_division=0)

        self.log_result("\n3. TF-IDF + Logistic Regression (Hyperparameter Tuned)")
        self.log_result("-" * 55)
        self.log_result(f"Best hyperparameters: {best_params}")
        self.log_result(f"Test Accuracy: {test_acc:.4f}")
        self.log_result(f"Test Macro F1: {test_f1_macro:.4f}")
        self.log_result(f"Test Micro F1: {test_f1_micro:.4f}")

        self.results["TF-IDF+LogReg"] = {
            "accuracy": test_acc, "macro_f1": test_f1_macro, "micro_f1": test_f1_micro,
            "best_params": best_params, "validation_f1": best_score
        }

    # ---------------- SimCSE ----------------

    def tune_simcse_experiment(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.log_hyperparams("\n🔧 SimCSE Hyperparameter Tuning")
        self.log_hyperparams("-" * 32)

        if not SIMCSE_AVAILABLE:
            self.log_result("\n❌ SimCSE not available (pip install simcse)")
            self.results["SimCSE"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "SimCSE not available"}
            return

        try:
            self.log_hyperparams("Loading SimCSE model...")
            simcse_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

            self.log_hyperparams("Encoding texts...")
            X_train_embeddings = simcse_model.encode(X_train)
            X_val_embeddings = simcse_model.encode(X_val)
            X_test_embeddings = simcse_model.encode(X_test)

            if hasattr(X_train_embeddings, 'cpu'):
                X_train_embeddings = X_train_embeddings.cpu().numpy()
                X_val_embeddings = X_val_embeddings.cpu().numpy()
                X_test_embeddings = X_test_embeddings.cpu().numpy()

            param_combinations = [
                (1.0, 'lbfgs', 1000, 'balanced'),
                (0.1, 'saga', 1000, 'balanced'),
                (10.0, 'liblinear', 1000, 'balanced'),
                (1.0, 'saga', 2000, None),
                (0.01, 'lbfgs', 1000, 'balanced'),
            ]

            best_score, best_params = 0, None
            self.log_hyperparams(f"Testing {len(param_combinations)} SimCSE classifier configurations...")

            for i, (C, solver, max_iter, class_weight) in enumerate(param_combinations):
                try:
                    clf = LogisticRegression(random_state=42, C=C, solver=solver, max_iter=max_iter, class_weight=class_weight)
                    clf.fit(X_train_embeddings, y_train)
                    y_pred_val = clf.predict(X_val_embeddings)

                    val_f1_macro = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
                    val_acc = accuracy_score(y_val, y_pred_val)

                    self.log_hyperparams(f"Config {i+1}: C={C}, solver={solver}, max_iter={max_iter}, class_weight={class_weight}")
                    self.log_hyperparams(f"   Validation F1: {val_f1_macro:.4f}, Accuracy: {val_acc:.4f}")

                    if val_f1_macro > best_score:
                        best_score = val_f1_macro
                        best_params = {'C': C, 'solver': solver, 'max_iter': max_iter, 'class_weight': class_weight}
                except Exception as e:
                    self.log_hyperparams(f"   Error: {e}")
                    continue

            self.log_hyperparams("\n🏆 Best SimCSE parameters:")
            for k, v in best_params.items():
                self.log_hyperparams(f"   {k}: {v}")
            self.log_hyperparams(f"   Validation F1: {best_score:.4f}")

            final_clf = LogisticRegression(random_state=42, **best_params)
            final_clf.fit(X_train_embeddings, y_train)
            y_pred = final_clf.predict(X_test_embeddings)

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

            self.log_result("\n4. SimCSE Classification (Hyperparameter Tuned)")
            self.log_result("-" * 47)
            self.log_result(f"Best hyperparameters: {best_params}")
            self.log_result(f"Test Accuracy: {acc:.4f}")
            self.log_result(f"Test Macro F1: {f1_macro:.4f}")
            self.log_result(f"Test Micro F1: {f1_micro:.4f}")

            self.results["SimCSE"] = {
                "accuracy": acc, "macro_f1": f1_macro, "micro_f1": f1_micro,
                "best_params": best_params, "validation_f1": best_score
            }

            report = classification_report(y_test, y_pred, zero_division=0)
            self.log_result(f"\nDetailed Classification Report:\n{report}")

        except Exception as e:
            self.log_result(f"❌ Error in SimCSE experiment: {str(e)}")
            self.results["SimCSE"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": str(e)}

    # ---------------- SBERT ----------------

    def tune_sentence_bert_experiment(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.log_hyperparams("\n🔧 Sentence-BERT Hyperparameter Tuning")
        self.log_hyperparams("-" * 37)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.log_result("\n❌ Sentence-Transformers not available (pip install sentence-transformers)")
            self.results["SBERT"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "SBERT not available"}
            return

        try:
            model_options = [
                'all-MiniLM-L6-v2',
                'all-mpnet-base-v2',
                'all-MiniLM-L12-v2',
            ]

            best_score, best_params, best_embeddings = 0, None, None

            for model_name in model_options:
                try:
                    self.log_hyperparams(f"\nTesting SBERT model: {model_name}")
                    sbert_model = SentenceTransformer(model_name)

                    X_train_emb = sbert_model.encode(X_train, show_progress_bar=False)
                    X_val_emb = sbert_model.encode(X_val, show_progress_bar=False)

                    classifier_configs = [
                        {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'class_weight': 'balanced'},
                        {'C': 0.1, 'solver': 'saga', 'max_iter': 1000, 'class_weight': 'balanced'},
                        {'C': 10.0, 'solver': 'liblinear', 'max_iter': 1000, 'class_weight': 'balanced'},
                        {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000, 'class_weight': None},
                    ]

                    for config in classifier_configs:
                        clf = LogisticRegression(random_state=42, **config)
                        clf.fit(X_train_emb, y_train)
                        y_pred_val = clf.predict(X_val_emb)

                        val_f1_macro = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
                        val_acc = accuracy_score(y_val, y_pred_val)

                        self.log_hyperparams(f"   Config {config}: F1={val_f1_macro:.4f}, Acc={val_acc:.4f}")

                        if val_f1_macro > best_score:
                            best_score = val_f1_macro
                            best_params = {'model_name': model_name, 'classifier_config': config}
                            X_test_emb = sbert_model.encode(X_test, show_progress_bar=False)
                            best_embeddings = {'train': X_train_emb, 'test': X_test_emb}

                except Exception as e:
                    self.log_hyperparams(f"   Error with {model_name}: {e}")
                    continue

            if best_params is None:
                self.log_result("\n❌ SBERT hyperparameter tuning failed")
                self.results["SBERT"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "Tuning failed"}
                return

            self.log_hyperparams(f"\n🏆 Best SBERT parameters:")
            self.log_hyperparams(f"   Model: {best_params['model_name']}")
            for k, v in best_params['classifier_config'].items():
                self.log_hyperparams(f"   {k}: {v}")
            self.log_hyperparams(f"   Validation F1: {best_score:.4f}")

            self.log_result("\n5. Sentence-BERT Classification (Hyperparameter Tuned)")
            self.log_result("-" * 57)

            final_classifier = LogisticRegression(random_state=42, **best_params['classifier_config'])
            final_classifier.fit(best_embeddings['train'], y_train)
            y_pred = final_classifier.predict(best_embeddings['test'])

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

            self.log_result(f"Best model: {best_params['model_name']}")
            self.log_result(f"Best classifier config: {best_params['classifier_config']}")
            self.log_result(f"Test Accuracy: {acc:.4f}")
            self.log_result(f"Test Macro F1: {f1_macro:.4f}")
            self.log_result(f"Test Micro F1: {f1_micro:.4f}")

            self.results["SBERT"] = {
                "accuracy": acc, "macro_f1": f1_macro, "micro_f1": f1_micro,
                "best_params": best_params, "validation_f1": best_score
            }

            report = classification_report(y_test, y_pred, zero_division=0)
            self.log_result(f"\nDetailed Classification Report:\n{report}")

        except Exception as e:
            self.log_result(f"❌ Error in SBERT experiment: {str(e)}")
            self.results["SBERT"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": str(e)}

    # ---------------- Transformers (BERT/RoBERTa) ----------------

    def run_transformer_experiments(self, X_train, X_val, X_test, y_train, y_val, y_test, labels):
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            self.log_result("\n❌ Skipping BERT/RoBERTa - dependencies not available")
            self.results["BERT"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "Deps not available"}
            self.results["RoBERTa"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "Deps not available"}
            return

        self._tune_and_run_bert(X_train, X_val, X_test, y_train, y_val, y_test, labels)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self._tune_and_run_roberta(X_train, X_val, X_test, y_train, y_val, y_test, labels)

    def _tune_and_run_bert(self, X_train, X_val, X_test, y_train, y_val, y_test, labels):
        self.log_hyperparams("\n🔧 BERT Hyperparameter Tuning")
        self.log_hyperparams("-" * 30)

        try:
            from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

            model_name = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(model_name)

            param_combinations = [
                (2e-5, 16, 5, 0.01, 256),
                (1e-5, 16, 3, 0.01, 256),
                (3e-5, 32, 8, 0.1, 512),
                (2e-5, 32, 5, 0.1, 256),
            ]

            best_score, best_params = 0, None
            self.log_hyperparams(f"Testing {len(param_combinations)} BERT parameter combinations...")

            for i, (lr, batch_size, epochs, weight_decay, max_length) in enumerate(param_combinations):
                try:
                    self.log_hyperparams(f"\nTesting BERT config {i+1}/{len(param_combinations)}: lr={lr}, batch={batch_size}, epochs={epochs}")

                    train_dataset = CustomDataset(X_train, y_train, tokenizer, max_length)
                    val_dataset = CustomDataset(X_val, y_val, tokenizer, max_length)

                    unique_labels = sorted(list(set(y_train)))
                    label2id = {label: idx for idx, label in enumerate(unique_labels)}
                    id2label = {idx: label for label, idx in label2id.items()}

                    model = BertForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=len(unique_labels),
                        problem_type="single_label_classification",
                        label2id=label2id,
                        id2label=id2label
                    ).to(self.device)

                    training_args = TrainingArguments(
                        output_dir=os.path.join(self.experiment_dir, f"bert_tune_{i}"),
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        warmup_steps=100,
                        weight_decay=weight_decay,
                        learning_rate=lr,
                        logging_steps=50,
                        evaluation_strategy="epoch",
                        save_strategy="no",
                        report_to=None,
                        seed=42,
                        dataloader_pin_memory=False,
                        fp16=True if self.device.type == "cuda" else False,
                        dataloader_num_workers=0,
                    )

                    def compute_metrics(eval_pred):
                        predictions, labels = eval_pred
                        predictions = np.argmax(predictions, axis=1)
                        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
                        return {"f1_macro": f1_macro}

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        compute_metrics=compute_metrics,
                    )

                    trainer.train()
                    eval_results = trainer.evaluate()
                    val_f1 = eval_results.get('eval_f1_macro', 0)
                    self.log_hyperparams(f"   Validation F1: {val_f1:.4f}")

                    if val_f1 > best_score:
                        best_score = val_f1
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'weight_decay': weight_decay,
                            'max_length': max_length
                        }

                    del model, trainer, train_dataset, val_dataset
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                except Exception as e:
                    self.log_hyperparams(f"   Error: {e}")
                    continue

            if best_params is None:
                self.results["BERT"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "Tuning failed"}
                self.log_result("\n❌ BERT hyperparameter tuning failed")
                return

            self.log_hyperparams("\n🏆 Best BERT parameters:")
            for k, v in best_params.items():
                self.log_hyperparams(f"   {k}: {v}")
            self.log_hyperparams(f"   Validation F1: {best_score:.4f}")

            # ========== Final training (train+val → test once) ==========
            self.log_result("\n6. BERT Classification (Hyperparameter Tuned)")
            self.log_result("-" * 45)
            self.log_result("Training final BERT model with best hyperparameters...")

            # Build combined train+val
            X_trval = X_train + X_val
            y_trval = y_train + y_val

            trainval_dataset = CustomDataset(X_trval, y_trval, tokenizer, best_params['max_length'])
            test_dataset = CustomDataset(X_test, y_test, tokenizer, best_params['max_length'])

            unique_labels = sorted(list(set(y_trval)))
            label2id = {label: idx for idx, label in enumerate(unique_labels)}
            id2label = {idx: label for label, idx in label2id.items()}
            y_test_numeric = [label2id[label] for label in y_test]

            final_model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(unique_labels),
                problem_type="single_label_classification",
                label2id=label2id,
                id2label=id2label
            ).to(self.device)

            final_training_args = TrainingArguments(
                output_dir=os.path.join(self.experiment_dir, "bert_final"),
                num_train_epochs=best_params['epochs'],
                per_device_train_batch_size=best_params['batch_size'],
                per_device_eval_batch_size=best_params['batch_size'],
                warmup_steps=100,
                weight_decay=best_params['weight_decay'],
                learning_rate=best_params['learning_rate'],
                logging_steps=50,
                evaluation_strategy="no",   # no validation here
                save_strategy="no",
                report_to=None,
                seed=42,
                dataloader_pin_memory=False,
                fp16=True if self.device.type == "cuda" else False,
                dataloader_num_workers=0,
            )

            final_trainer = Trainer(
                model=final_model,
                args=final_training_args,
                train_dataset=trainval_dataset,
                eval_dataset=None,
            )

            train_start = time.time()
            final_trainer.train()
            train_time = time.time() - train_start

            predictions = final_trainer.predict(test_dataset)
            y_pred_numeric = np.argmax(predictions.predictions, axis=1)

            final_acc = accuracy_score(y_test_numeric, y_pred_numeric)
            final_f1_macro = f1_score(y_test_numeric, y_pred_numeric, average='macro', zero_division=0)
            final_f1_micro = f1_score(y_test_numeric, y_pred_numeric, average='micro', zero_division=0)

            self.log_result(f"Best hyperparameters: {best_params}")
            self.log_result(f"Training time: {train_time:.2f} seconds")
            self.log_result(f"Test Accuracy: {final_acc:.4f}")
            self.log_result(f"Test Macro F1: {final_f1_macro:.4f}")
            self.log_result(f"Test Micro F1: {final_f1_micro:.4f}")

            self.results["BERT"] = {
                "accuracy": final_acc, "macro_f1": final_f1_macro, "micro_f1": final_f1_micro,
                "best_params": best_params, "validation_f1": best_score, "training_time": train_time
            }

            del final_model, final_trainer, trainval_dataset, test_dataset
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            self.log_result(f"❌ Error in BERT experiment: {str(e)}")
            self.results["BERT"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": str(e)}

    def _tune_and_run_roberta(self, X_train, X_val, X_test, y_train, y_val, y_test, labels):
        self.log_hyperparams("\n🔧 RoBERTa Hyperparameter Tuning")
        self.log_hyperparams("-" * 33)

        try:
            from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer

            model_name = "roberta-base"
            tokenizer = RobertaTokenizer.from_pretrained(model_name)

            param_combinations = [
                (2e-5, 16, 5, 0.01, 256),
                (1e-5, 16, 3, 0.01, 256),
                (3e-5, 32, 8, 0.1, 512),
                (2e-5, 32, 5, 0.1, 256),
            ]

            best_score, best_params = 0, None
            self.log_hyperparams(f"Testing {len(param_combinations)} RoBERTa parameter combinations...")

            for i, (lr, batch_size, epochs, weight_decay, max_length) in enumerate(param_combinations):
                try:
                    self.log_hyperparams(f"\nTesting RoBERTa config {i+1}/{len(param_combinations)}: lr={lr}, batch={batch_size}, epochs={epochs}")

                    train_dataset = CustomDataset(X_train, y_train, tokenizer, max_length)
                    val_dataset = CustomDataset(X_val, y_val, tokenizer, max_length)

                    unique_labels = sorted(list(set(y_train)))
                    label2id = {label: idx for idx, label in enumerate(unique_labels)}
                    id2label = {idx: label for label, idx in label2id.items()}

                    model = RobertaForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=len(unique_labels),
                        problem_type="single_label_classification",
                        label2id=label2id,
                        id2label=id2label
                    ).to(self.device)

                    training_args = TrainingArguments(
                        output_dir=os.path.join(self.experiment_dir, f"roberta_tune_{i}"),
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        warmup_steps=100,
                        weight_decay=weight_decay,
                        learning_rate=lr,
                        logging_steps=50,
                        evaluation_strategy="epoch",
                        save_strategy="no",
                        report_to=None,
                        seed=42,
                        dataloader_pin_memory=False,
                        fp16=True if self.device.type == "cuda" else False,
                        dataloader_num_workers=0,
                    )

                    def compute_metrics(eval_pred):
                        predictions, labels = eval_pred
                        predictions = np.argmax(predictions, axis=1)
                        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
                        return {"f1_macro": f1_macro}

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        compute_metrics=compute_metrics,
                    )

                    trainer.train()
                    eval_results = trainer.evaluate()
                    val_f1 = eval_results.get('eval_f1_macro', 0)
                    self.log_hyperparams(f"   Validation F1: {val_f1:.4f}")

                    if val_f1 > best_score:
                        best_score = val_f1
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'weight_decay': weight_decay,
                            'max_length': max_length
                        }

                    del model, trainer, train_dataset, val_dataset
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                except Exception as e:
                    self.log_hyperparams(f"   Error: {e}")
                    continue

            if best_params is None:
                self.results["RoBERTa"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": "Tuning failed"}
                self.log_result("\n❌ RoBERTa hyperparameter tuning failed")
                return

            self.log_hyperparams("\n🏆 Best RoBERTa parameters:")
            for k, v in best_params.items():
                self.log_hyperparams(f"   {k}: {v}")
            self.log_hyperparams(f"   Validation F1: {best_score:.4f}")

            # ========== Final training (train+val → test once) ==========
            self.log_result("\n7. RoBERTa Classification (Hyperparameter Tuned)")
            self.log_result("-" * 48)
            self.log_result("Training final RoBERTa model with best hyperparameters...")

            X_trval = X_train + X_val
            y_trval = y_train + y_val

            trainval_dataset = CustomDataset(X_trval, y_trval, tokenizer, best_params['max_length'])
            test_dataset = CustomDataset(X_test, y_test, tokenizer, best_params['max_length'])

            unique_labels = sorted(list(set(y_trval)))
            label2id = {label: idx for idx, label in enumerate(unique_labels)}
            id2label = {idx: label for label, idx in label2id.items()}
            y_test_numeric = [label2id[label] for label in y_test]

            final_model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(unique_labels),
                problem_type="single_label_classification",
                label2id=label2id,
                id2label=id2label
            ).to(self.device)

            final_training_args = TrainingArguments(
                output_dir=os.path.join(self.experiment_dir, "roberta_final"),
                num_train_epochs=best_params['epochs'],
                per_device_train_batch_size=best_params['batch_size'],
                per_device_eval_batch_size=best_params['batch_size'],
                warmup_steps=100,
                weight_decay=best_params['weight_decay'],
                learning_rate=best_params['learning_rate'],
                logging_steps=50,
                evaluation_strategy="no",
                save_strategy="no",
                report_to=None,
                seed=42,
                dataloader_pin_memory=False,
                fp16=True if self.device.type == "cuda" else False,
                dataloader_num_workers=0,
            )

            final_trainer = Trainer(
                model=final_model,
                args=final_training_args,
                train_dataset=trainval_dataset,
                eval_dataset=None,
            )

            train_start = time.time()
            final_trainer.train()
            train_time = time.time() - train_start

            predictions = final_trainer.predict(test_dataset)
            y_pred_numeric = np.argmax(predictions.predictions, axis=1)

            final_acc = accuracy_score(y_test_numeric, y_pred_numeric)
            final_f1_macro = f1_score(y_test_numeric, y_pred_numeric, average='macro', zero_division=0)
            final_f1_micro = f1_score(y_test_numeric, y_pred_numeric, average='micro', zero_division=0)

            self.log_result(f"Best hyperparameters: {best_params}")
            self.log_result(f"Training time: {train_time:.2f} seconds")
            self.log_result(f"Test Accuracy: {final_acc:.4f}")
            self.log_result(f"Test Macro F1: {final_f1_macro:.4f}")
            self.log_result(f"Test Micro F1: {final_f1_micro:.4f}")

            self.results["RoBERTa"] = {
                "accuracy": final_acc, "macro_f1": final_f1_macro, "micro_f1": final_f1_micro,
                "best_params": best_params, "validation_f1": best_score, "training_time": train_time
            }

            del final_model, final_trainer, trainval_dataset, test_dataset
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            self.log_result(f"❌ Error in RoBERTa experiment: {str(e)}")
            self.results["RoBERTa"] = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "error": str(e)}

    # ---------------- Summary ----------------

    def create_results_summary(self):
        self.log_result("\n" + "="*80)
        self.log_result("EXPERIMENT SUMMARY")
        self.log_result("="*80)

        valid_results = {}
        for model, metrics in self.results.items():
            if isinstance(metrics.get("macro_f1"), (int, float)) and metrics["macro_f1"] > 0:
                valid_results[model] = metrics

        if not valid_results:
            self.log_result("❌ No successful experiments to summarize")
            return

        results_df = pd.DataFrame(valid_results).T
        columns_to_keep = ['accuracy', 'macro_f1', 'micro_f1']
        for extra in ['validation_f1', 'training_time']:
            if extra in results_df.columns:
                columns_to_keep.append(extra)

        results_df = results_df[columns_to_keep].round(4)

        self.log_result("\nResults Table:")
        self.log_result(results_df.to_string())

        csv_path = os.path.join(self.experiment_dir, "results_summary.csv")
        results_df.to_csv(csv_path)
        self.log_result(f"\n📊 Results saved to: {csv_path}")

        if len(results_df) > 0:
            best_macro_f1 = results_df['macro_f1'].max()
            best_model = results_df['macro_f1'].idxmax()
            self.log_result(f"\n🏆 Best performing model: {best_model}")
            self.log_result(f"   Test Macro F1: {best_macro_f1:.4f}")
            self.log_result(f"   Test Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")
            if 'validation_f1' in results_df.columns:
                self.log_result(f"   Validation F1: {results_df.loc[best_model, 'validation_f1']:.4f}")


def run_all_experiments(datasets: List[str], split_dir: str = "split_datasets_enriched_2"):
    print("🚀  TEXT CLASSIFICATION EXPERIMENTS")
    print("="*80)
    print("Using Train/Validation/Test Split (60:20:20)")
    print("With hyperparameter tuning for training-based methods")
    print(f"Available: PyTorch={TORCH_AVAILABLE}, Transformers={TRANSFORMERS_AVAILABLE}")
    print(f"SimCSE={SIMCSE_AVAILABLE}, SBERT={SENTENCE_TRANSFORMERS_AVAILABLE}")
    print("="*80)

    all_results = {}
    experiment_folders = []

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"🚀 RUNNING EXPERIMENTS ON: {dataset_name.upper()}")
        print(f"{'='*80}")

        try:
            processor = SplitDatasetProcessor(dataset_name, split_dir)
            train_users, val_users, test_users, labels = processor.load_split_data()

            X_train, y_train, _ = processor.create_text_features(train_users)
            X_val, y_val, _ = processor.create_text_features(val_users)
            X_test, y_test, _ = processor.create_text_features(test_users)

            if len(X_train) < 10 or len(X_val) < 5 or len(X_test) < 5:
                print(f"❌ Insufficient data for {dataset_name}: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
                continue

            print(f"📊 Dataset splits: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
            print(f"📋 Classes: {len(labels)} ({labels})")

            experiments = TextClassificationExperiments(dataset_name)
            experiment_folders.append(experiments.experiment_dir)

            experiments.run_baseline_experiments(X_train, X_val, X_test, y_train, y_val, y_test)
            experiments.tune_tfidf_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test)
            experiments.tune_simcse_experiment(X_train, X_val, X_test, y_train, y_val, y_test)
            experiments.tune_sentence_bert_experiment(X_train, X_val, X_test, y_train, y_val, y_test)
            experiments.run_transformer_experiments(X_train, X_val, X_test, y_train, y_val, y_test, labels)

            experiments.create_results_summary()
            all_results[dataset_name] = experiments.results

        except Exception as e:
            print(f"❌ Error with dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Overall summary files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_dir = f"overall_text_experiments_{timestamp}"
    os.makedirs(overall_dir, exist_ok=True)

    summary_file = os.path.join(overall_dir, "overall_summary.txt")
    hyperparams_summary_file = os.path.join(overall_dir, "hyperparameters_summary.txt")

    with open(summary_file, "w") as f:
        f.write(" TEXT CLASSIFICATION EXPERIMENTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Using Train/Validation/Test Split (60:20:20)\n")
        f.write("With hyperparameter tuning for training-based methods\n")
        f.write(f"Dependencies: PyTorch={TORCH_AVAILABLE}, Transformers={TRANSFORMERS_AVAILABLE}\n")
        f.write(f"SimCSE={SIMCSE_AVAILABLE}, SBERT={SENTENCE_TRANSFORMERS_AVAILABLE}\n")
        f.write("="*80 + "\n\n")
        for dataset, results in all_results.items():
            f.write(f"\n{dataset.upper()}:\n")
            f.write("-" * 40 + "\n")
            for model, metrics in results.items():
                if isinstance(metrics.get("macro_f1"), (int, float)) and metrics["macro_f1"] > 0:
                    f.write(f"{model:<15} Test Macro F1: {metrics['macro_f1']:.4f} | Test Accuracy: {metrics['accuracy']:.4f}")
                    if 'validation_f1' in metrics:
                        f.write(f" | Val F1: {metrics['validation_f1']:.4f}")
                    f.write("\n")
                elif 'error' in metrics:
                    f.write(f"{model:<15} Error: {metrics['error']}\n")
                else:
                    f.write(f"{model:<15} Skipped\n")

    with open(hyperparams_summary_file, "w") as f:
        f.write("HYPERPARAMETER SEARCH SUMMARY\n")
        f.write("="*50 + "\n\n")
        for dataset, results in all_results.items():
            f.write(f"\n{dataset.upper()} - Best Hyperparameters:\n")
            f.write("-" * 40 + "\n")
            for model, metrics in results.items():
                if 'best_params' in metrics:
                    f.write(f"{model}:\n")
                    for param, value in metrics['best_params'].items():
                        f.write(f"  {param}: {value}\n")
                    if 'validation_f1' in metrics:
                        f.write(f"  validation_f1: {metrics['validation_f1']:.4f}\n")
                    f.write("\n")

    print(f"\n🎉 EXPERIMENTS COMPLETE!")
    print(f"📁 Overall summary: {overall_dir}")
    print(f"📁 Individual experiment folders: {len(experiment_folders)}")
    print(f"📋 Hyperparameters summary: {hyperparams_summary_file}")

    return all_results, experiment_folders


def main():
    print("🔧 DEPENDENCY STATUS:")
    print(f"   PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"   Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌ (try: pip install torch transformers)'}")
    print(f"   SimCSE: {'✅' if SIMCSE_AVAILABLE else '❌ (try: pip install simcse)'}")
    print(f"   SBERT: {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌ (try: pip install sentence-transformers)'}")
    print()

    # Default datasets and split dir
    DATASETS = ["abortion", "trump", "religion"]
    SPLIT_DIR = "split_datasets_enriched_2"

    # CLI parsing
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage:")
            print("  python text_classification_experiments.py                    # Run all datasets")
            print("  python text_classification_experiments.py dataset1 dataset2  # Run specific datasets")
            print("  python text_classification_experiments.py --split_dir path   # Use custom split directory")
            return
        else:
            args = sys.argv[1:]
            if "--split_dir" in args:
                i = args.index("--split_dir")
                if i + 1 < len(args):
                    SPLIT_DIR = args[i + 1]
                    args = args[:i] + args[i + 2:]
            if args:
                DATASETS = args

    print(f"📁 Split directory: {SPLIT_DIR}")
    print(f"📊 Datasets: {DATASETS}")

    # Check split dirs
    missing_paths = []
    for dataset in DATASETS:
        dataset_dir = os.path.join(SPLIT_DIR, dataset)
        if not os.path.exists(dataset_dir):
            missing_paths.append(dataset_dir)
            continue
        for fname in ["train.json", "validation.json", "test.json"]:
            fpath = os.path.join(dataset_dir, fname)
            if not os.path.exists(fpath):
                missing_paths.append(fpath)

    if missing_paths:
        print("❌ Missing split files or directories:")
        for p in missing_paths:
            print(f"   • {p}")
        print("\n💡 Make sure you have run split_dataset.py first to create the splits:")
        print(f"   python split_dataset.py --dataset <dataset_name> --output_dir {SPLIT_DIR}")
        print("\n   Expected structure:")
        print(f"   {SPLIT_DIR}/")
        print(f"   ├── dataset1/")
        print(f"   │   ├── train.json")
        print(f"   │   ├── validation.json")
        print(f"   │   ├── test.json")
        print(f"   │   └── metadata.json")
        print(f"   └── dataset2/...")
        return

    set_seed(42)

    print("\n🔥 Starting experiments with hyperparameter tuning...")
    print("Methods include:")
    print("  • Baselines: Random, Majority (no tuning)")
    print("  • TF-IDF + LogReg: Tuning max_features, ngram_range, C, solver")
    print("  • SimCSE: Tuning classifier head (C, solver, max_iter, class_weight)")
    print("  • Sentence-BERT: Model selection + classifier head tuning")
    print("  • BERT: Full fine-tuning (lr, batch_size, epochs, weight_decay, max_length)")
    print("  • RoBERTa: Full fine-tuning (lr, batch_size, epochs, weight_decay, max_length)")

    try:
        all_results, experiment_folders = run_all_experiments(DATASETS, SPLIT_DIR)

        print("\n✅ Experiments completed successfully!")
        print("\n📋 Results locations:")
        for folder in experiment_folders:
            print(f"   • {folder}/")
            print(f"     ├── results.txt (detailed logs)")
            print(f"     ├── hyperparameter_search.txt (tuning logs)")
            print(f"     └── results_summary.csv (metrics)")

    except KeyboardInterrupt:
        print("\n⚠️  Experiments interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
