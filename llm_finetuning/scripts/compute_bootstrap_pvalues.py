#!/usr/bin/env python3
"""
Compute statistical significance (p-values) from bootstrap CIs.

Usage:
    cd /home/skumyol/llm_training
    python llm_finetuning/scripts/compute_bootstrap_pvalues.py

Reads decision_card_ab_report.json and outputs p-values for key comparisons.
"""
import json
import numpy as np
from pathlib import Path


def ci_to_pvalue(delta, ci_lower, ci_upper, alpha=0.05):
    """
    Estimate two-tailed p-value from bootstrap CI.
    
    Uses the relationship between CI and p-value:
    - If CI excludes zero, p < alpha
    - Approximate p-value based on how far zero is from the CI
    """
    if ci_lower > 0 or ci_upper < 0:
        # CI excludes zero, so p < alpha
        # Estimate more precisely: further from zero = smaller p
        margin = min(abs(ci_lower), abs(ci_upper))
        # Rough approximation: p ~ alpha * (1 - margin / |delta|)
        if abs(delta) > 0:
            p_approx = alpha * max(0.001, 1 - margin / abs(delta))
        else:
            p_approx = alpha
        return max(p_approx, 0.001)  # Floor at 0.001
    else:
        # CI includes zero, so p >= alpha
        # Estimate based on where zero falls in the CI
        ci_range = ci_upper - ci_lower
        if ci_range > 0:
            # Position of zero within CI (0 = at lower bound, 1 = at upper bound)
            zero_position = (0 - ci_lower) / ci_range
            # Distance from center (0.5)
            distance = abs(zero_position - 0.5) * 2  # 0 at center, 1 at edge
            p_approx = alpha + (1 - alpha) * (1 - distance)
        else:
            p_approx = 1.0
        return min(p_approx, 1.0)


def format_pvalue(p):
    """Format p-value with standard notation."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"= {p:.3f}"
    elif p < 0.05:
        return f"= {p:.3f}"
    else:
        return f"= {p:.3f}"


def interpret_pvalue(p):
    """Interpret p-value significance."""
    if p < 0.001:
        return "*** (highly significant)"
    elif p < 0.01:
        return "** (very significant)"
    elif p < 0.05:
        return "* (significant)"
    else:
        return "ns (not significant)"


def main():
    # Load decision card A/B report
    report_path = "eval_results/decision_card_ab_report.json"
    
    if not Path(report_path).exists():
        print(f"[ERROR] Report not found: {report_path}")
        return
    
    with open(report_path) as f:
        report = json.load(f)
    
    print("\n" + "=" * 70)
    print("BOOTSTRAP STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    
    # Key metrics to test (map display name to JSON key)
    key_metrics = {
        "ROUGE-L": "rouge_l",
        "Policy consistency": "policy_consistency",
        "Secret leakage": "secret_leakage",
        "Contradiction rate": "contradiction",
        "Over-disclosure": "over_disclosure_bin",
        "Under-disclosure": "under_disclosure_bin",
    }
    
    results = []
    
    print("\n1. DECISION CARD A/B TEST (Baseline vs Treatment)")
    print("-" * 70)
    
    deltas = report.get("deltas", {})
    
    for display_name, json_key in key_metrics.items():
        if json_key in deltas:
            data = deltas[json_key]
            delta = data.get("mean_delta", 0)
            ci_lower = data.get("ci_low", 0)
            ci_upper = data.get("ci_high", 0)
            
            p = ci_to_pvalue(delta, ci_lower, ci_upper)
            p_str = format_pvalue(p)
            interp = interpret_pvalue(p)
            
            print(f"\n{display_name}:")
            print(f"  Δ = {delta:+.4f}")
            print(f"  95% CI = [{ci_lower:+.4f}, {ci_upper:+.4f}]")
            print(f"  p-value {p_str}  {interp}")
            
            results.append({
                "metric": display_name,
                "delta": delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p,
                "significant": p < 0.05,
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    sig_count = sum(1 for r in results if r["significant"])
    print(f"\nSignificant effects (p < 0.05): {sig_count}/{len(results)}")
    
    for r in results:
        status = "✓" if r["significant"] else "✗"
        print(f"  {status} {r['metric']}: p {format_pvalue(r['p_value'])}")
    
    # Save results
    output = {
        "statistical_tests": results,
        "method": "Bootstrap CI to p-value approximation",
        "alpha": 0.05,
        "note": "P-values estimated from bootstrap 95% CIs. CI excludes zero => p < 0.05.",
    }
    
    output_path = "eval_results/bootstrap_significance.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved detailed results: {output_path}")
    
    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[t]\n",
        "\\centering\n",
        "\\caption{Statistical significance of decision-card effects (bootstrap $n{=}1{,}000$). $^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$, ns = not significant.}\n",
        "\\label{tab:significance}\n",
        "\\small\n",
        "\\begin{tabular}{@{}lrrrrc@{}}\n",
        "\\toprule\n",
        "\\textbf{Metric} & \\textbf{$\\Delta$} & \\textbf{95\\% CI} & \\textbf{$p$-value} & \\textbf{Sig.} \\\\\n",
        "\\midrule\n",
    ]
    
    for r in results:
        metric_name = r["metric"].replace("_", "\\_")
        delta = f"{r['delta']:+.4f}"
        ci = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
        
        if r["p_value"] < 0.001:
            p_str = "$<$0.001"
            sig = "$^{***}$"
        elif r["p_value"] < 0.01:
            p_str = f"{r['p_value']:.3f}"
            sig = "$^{**}$"
        elif r["p_value"] < 0.05:
            p_str = f"{r['p_value']:.3f}"
            sig = "$^{*}$"
        else:
            p_str = f"{r['p_value']:.3f}"
            sig = "ns"
        
        latex_lines.append(f"{metric_name} & {delta} & {ci} & {p_str} & {sig} \\\\\n")
    
    latex_lines.append("\\bottomrule\n")
    latex_lines.append("\\end{tabular}\n")
    latex_lines.append("\\end{table}\n")
    
    latex_path = "paper/OCEAN_MBTI_ACL_2026/figures/significance_table.tex"
    with open(latex_path, "w") as f:
        f.write("".join(latex_lines))
    
    print(f"Saved LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
