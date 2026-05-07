from src.eval.eval_response import (
    _corpus_bleu,
    _degenerate_repetition_rate,
    _prompt_artifact_rate,
    _repeated_ngram_rate,
)
from src.metrics_report import compute_latent_metrics


def test_latent_metrics_include_agreement_statistics():
    metrics = compute_latent_metrics(
        all_preds={"response_policy": [0, 1, 1, 2], "reveal_decision": [0, 1, 2, 2]},
        all_golds={"response_policy": [0, 1, 2, 2], "reveal_decision": [0, 1, 1, 2]},
    )

    response_policy = metrics["fields"]["response_policy"]
    assert "balanced_accuracy" in response_policy
    assert "cohen_kappa" in response_policy
    assert "mcc" in response_policy
    assert metrics["summary"]["mean_cohen_kappa"] > 0.0


def test_response_eval_detects_degenerate_generation():
    text = "Can you stand? Can you stand? Can you stand?"
    assert _repeated_ngram_rate(text, n=3) > 0.25
    assert _degenerate_repetition_rate([text], n=3, threshold=0.25) == 1.0


def test_response_eval_detects_prompt_artifacts():
    outputs = [
        "Based on the player's question, the response should be cautious.",
        "<response><speech>I cannot say.</speech></response>",
        "I cannot say.",
    ]
    assert _prompt_artifact_rate(outputs) == 2 / 3


def test_corpus_bleu_rewards_exact_matches():
    references = ["I can repair your blade for a fair price."]
    assert _corpus_bleu(references, references, max_n=4) > 0.99
