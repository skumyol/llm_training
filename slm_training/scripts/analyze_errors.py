#!/usr/bin/env python3
"""Error analysis for personality and affect encoders."""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from src.models.personality import DistilBertRegressor as PersonalityModel
from src.models.affect import DistilBertRegressor as AffectModel
from src.data.datasets import PersonalityDataset, AffectDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def analyze_personality_errors(
    model_path: str,
    val_path: str = "data/personality/val.csv",
    output_dir: str = "artifacts/analysis"
):
    """Analyze personality prediction errors by trait."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PersonalityModel("distilbert-base-uncased", out_dim=5, dropout=0.2, use_sigmoid=True)
    ckpt = torch.load(Path(model_path) / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    df = pd.read_csv(val_path)
    trait_cols = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    
    # Get predictions
    dataset = PersonalityDataset(df, tokenizer, trait_cols, max_length=256)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds, all_labels, all_texts = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            preds = (torch.sigmoid(outputs["logits"]) > 0.5).cpu().int()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Per-trait analysis
    results = {}
    for i, trait in enumerate(trait_cols):
        tn, fp, fn, tp = confusion_matrix(all_labels[:, i], all_preds[:, i]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[trait] = {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn
        }
    
    # Print analysis
    print("\n" + "="*60)
    print("PERSONALITY ERROR ANALYSIS")
    print("="*60)
    for trait, metrics in results.items():
        print(f"\n{trait.upper()}:")
        print(f"  F1: {metrics['f1']:.3f} | Acc: {metrics['accuracy']:.3f}")
        print(f"  FP: {metrics['false_positives']} | FN: {metrics['false_negatives']}")
        
        # Identify which traits are hardest
        if metrics['f1'] < 0.6:
            print(f"  ⚠️  LOW F1 - needs improvement")
    
    return results

def analyze_affect_errors(
    model_path: str,
    val_path: str = "data/affect/val.csv",
    output_dir: str = "artifacts/analysis"
):
    """Analyze affect prediction errors by dimension and value range."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = AffectModel("distilbert-base-uncased", out_dim=3, dropout=0.1)
    ckpt = torch.load(Path(model_path) / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    df = pd.read_csv(val_path)
    target_cols = ["valence", "arousal", "dominance"]
    
    dataset = AffectDataset(df, tokenizer, target_cols, max_length=256)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            preds = outputs["preds"].cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Per-dimension analysis
    print("\n" + "="*60)
    print("AFFECT ERROR ANALYSIS")
    print("="*60)
    
    results = {}
    for i, dim in enumerate(target_cols):
        errors = all_preds[:, i] - all_labels[:, i]
        mae = np.abs(errors).mean()
        mse = (errors ** 2).mean()
        
        # Error by true value range
        low_mask = all_labels[:, i] < 0.33
        mid_mask = (all_labels[:, i] >= 0.33) & (all_labels[:, i] < 0.67)
        high_mask = all_labels[:, i] >= 0.67
        
        results[dim] = {
            "mae": mae,
            "mse": mse,
            "mean_error": errors.mean(),
            "std_error": errors.std(),
            "low_range_mae": np.abs(errors[low_mask]).mean() if low_mask.any() else 0,
            "mid_range_mae": np.abs(errors[mid_mask]).mean() if mid_mask.any() else 0,
            "high_range_mae": np.abs(errors[high_mask]).mean() if high_mask.any() else 0,
        }
        
        print(f"\n{dim.upper()}:")
        print(f"  Overall MAE: {mae:.4f}")
        print(f"  Bias: {errors.mean():+.4f} (positive = overestimating)")
        print(f"  Low values (0-0.33) MAE: {results[dim]['low_range_mae']:.4f}")
        print(f"  Mid values (0.33-0.67) MAE: {results[dim]['mid_range_mae']:.4f}")
        print(f"  High values (0.67-1.0) MAE: {results[dim]['high_range_mae']:.4f}")
        
        # Identify range with most errors
        worst_range = max([("low", results[dim]['low_range_mae']), 
                          ("mid", results[dim]['mid_range_mae']),
                          ("high", results[dim]['high_range_mae'])], 
                         key=lambda x: x[1])
        print(f"  ⚠️  Worst performance in {worst_range[0]} value range")
    
    return results

if __name__ == "__main__":
    import sys
    
    Path("artifacts/analysis").mkdir(parents=True, exist_ok=True)
    
    print("Running error analysis...")
    
    # Analyze personality
    personality_results = analyze_personality_errors(
        "artifacts/personality_encoder/personality_v2_aggressive/best_model"
    )
    
    # Analyze affect
    affect_results = analyze_affect_errors(
        "artifacts/affect_encoder/affect_v5_ccc_combo/best_model"
    )
    
    # Save results
    with open("artifacts/analysis/error_analysis.json", "w") as f:
        json.dump({
            "personality": personality_results,
            "affect": affect_results
        }, f, indent=2)
    
    print("\n✓ Analysis complete. Results saved to artifacts/analysis/error_analysis.json")
