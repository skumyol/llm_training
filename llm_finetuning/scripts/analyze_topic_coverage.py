#!/usr/bin/env python3
"""Analyze topic coverage and collapse in generated responses.

Primary mode uses BERTopic when installed. If BERTopic is unavailable, the
script falls back to a lightweight TF-IDF + NMF topic model from scikit-learn.
The output is diagnostic: use it to identify topic collapse or generic refusal
clusters, not as a primary quality metric.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _load_texts(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        samples = json.load(f)
    gold = [s.get("gold", "").strip() for s in samples if s.get("gold") and s.get("generated")]
    generated = [s.get("generated", "").strip() for s in samples if s.get("gold") and s.get("generated")]
    return gold, generated


def _entropy(labels: list[int]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())


def _top_counts(labels: list[int], k: int = 10) -> list[dict]:
    total = max(1, len(labels))
    return [
        {"topic": int(topic), "count": int(count), "rate": count / total}
        for topic, count in Counter(labels).most_common(k)
    ]


def _bertopic_analysis(gold: list[str], generated: list[str], n_topics: int | None) -> dict:
    from bertopic import BERTopic

    docs = gold + generated
    model = BERTopic(nr_topics=n_topics, calculate_probabilities=False, verbose=False)
    labels, _ = model.fit_transform(docs)
    gold_labels = labels[: len(gold)]
    gen_labels = labels[len(gold) :]
    topic_info = model.get_topic_info()
    topics = []
    for _, row in topic_info.head(20).iterrows():
        topic = int(row["Topic"])
        if topic == -1:
            words = "outlier"
        else:
            words = ", ".join(word for word, _ in model.get_topic(topic)[:8])
        topics.append({"topic": topic, "count": int(row["Count"]), "words": words})
    return {
        "method": "bertopic",
        "n_gold": len(gold),
        "n_generated": len(generated),
        "gold_topic_entropy": _entropy(gold_labels),
        "generated_topic_entropy": _entropy(gen_labels),
        "gold_outlier_rate": sum(t == -1 for t in gold_labels) / max(1, len(gold_labels)),
        "generated_outlier_rate": sum(t == -1 for t in gen_labels) / max(1, len(gen_labels)),
        "gold_top_topics": _top_counts(gold_labels),
        "generated_top_topics": _top_counts(gen_labels),
        "topic_descriptions": topics,
    }


def _nmf_analysis(gold: list[str], generated: list[str], n_topics: int) -> dict:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = gold + generated
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, stop_words="english")
    x = vectorizer.fit_transform(docs)
    n_topics = min(n_topics, max(2, x.shape[0] - 1), max(2, x.shape[1] - 1))
    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    topic_weights = model.fit_transform(x)
    labels = topic_weights.argmax(axis=1).tolist()
    gold_labels = labels[: len(gold)]
    gen_labels = labels[len(gold) :]
    names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, weights in enumerate(model.components_):
        top = weights.argsort()[-8:][::-1]
        topics.append({"topic": topic_idx, "words": ", ".join(names[i] for i in top)})
    return {
        "method": "tfidf_nmf_fallback",
        "n_gold": len(gold),
        "n_generated": len(generated),
        "gold_topic_entropy": _entropy(gold_labels),
        "generated_topic_entropy": _entropy(gen_labels),
        "gold_top_topics": _top_counts(gold_labels),
        "generated_top_topics": _top_counts(gen_labels),
        "topic_descriptions": topics,
    }


def _lexical_fallback(gold: list[str], generated: list[str], error: Exception) -> dict:
    def top_terms(texts: list[str]) -> list[dict]:
        stop = {
            "the", "and", "that", "you", "for", "with", "this", "not", "but",
            "are", "can", "will", "your", "have", "from", "they", "our",
        }
        counts = Counter()
        for text in texts:
            for tok in text.lower().split():
                tok = "".join(ch for ch in tok if ch.isalnum())
                if len(tok) >= 4 and tok not in stop:
                    counts[tok] += 1
        total = max(1, sum(counts.values()))
        return [{"term": term, "count": count, "rate": count / total} for term, count in counts.most_common(30)]

    return {
        "method": "lexical_fallback",
        "n_gold": len(gold),
        "n_generated": len(generated),
        "gold_top_terms": top_terms(gold),
        "generated_top_terms": top_terms(generated),
        "topic_model_skipped": str(error),
    }


def analyze(input_path: str, output_path: str, n_topics: int) -> None:
    gold, generated = _load_texts(input_path)
    try:
        result = _bertopic_analysis(gold, generated, n_topics)
    except Exception as bertopic_error:
        try:
            result = _nmf_analysis(gold, generated, n_topics)
            result["bertopic_skipped"] = str(bertopic_error)
        except Exception as fallback_error:
            result = _lexical_fallback(gold, generated, fallback_error)
            result["bertopic_skipped"] = str(bertopic_error)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Topic coverage written to {out}")
    print(f"Method: {result['method']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated-vs-gold topic coverage")
    parser.add_argument("--input", default="eval_results/sample_generations.json")
    parser.add_argument("--output", default="eval_results/topic_coverage.json")
    parser.add_argument("--n-topics", type=int, default=12)
    args = parser.parse_args()
    analyze(args.input, args.output, args.n_topics)


if __name__ == "__main__":
    main()
