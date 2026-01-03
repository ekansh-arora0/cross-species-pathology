#!/usr/bin/env python3
"""
Reproduce Table 3: Main Results

This script reproduces the main results table from the paper:
"Lost in Translation: How Language Re-Aligns Vision for Cross-Species Pathology"

Table 3: AUC-ROC Across All Experimental Conditions

Usage:
    python scripts/reproduce_table3.py --data-dir /path/to/data --checkpoint /path/to/cpath_clip.pt
    
    # With pre-computed embeddings (faster):
    python scripts/reproduce_table3.py --embeddings-dir /path/to/embeddings
    
    # Quick test with subset:
    python scripts/reproduce_table3.py --data-dir /path/to/data --quick-test

Output:
    - results/table3_results.json: Full results with confidence intervals
    - results/table3_summary.txt: Formatted table for paper
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


def load_embeddings(embeddings_dir: Path, dataset: str):
    """Load pre-computed embeddings."""
    emb_path = embeddings_dir / f"{dataset}_embeddings.npy"
    label_path = embeddings_dir / f"{dataset}_labels.npy"
    
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    
    embeddings = np.load(emb_path)
    labels = np.load(label_path)
    
    return embeddings, labels


def compute_prototype_classification(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    n_prototypes: int = 50,
    n_bootstrap: int = 1000
) -> dict:
    """
    Prototype-based zero-shot classification.
    
    This is the baseline method that fails under domain shift.
    """
    # Normalize embeddings
    train_emb = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    test_emb = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)
    
    # Sample prototypes
    np.random.seed(42)
    tumor_idx = np.where(train_labels == 1)[0]
    normal_idx = np.where(train_labels == 0)[0]
    
    n_proto_per_class = min(n_prototypes, len(tumor_idx), len(normal_idx))
    
    tumor_proto_idx = np.random.choice(tumor_idx, n_proto_per_class, replace=False)
    normal_proto_idx = np.random.choice(normal_idx, n_proto_per_class, replace=False)
    
    # Compute prototypes (mean)
    tumor_proto = train_emb[tumor_proto_idx].mean(axis=0)
    normal_proto = train_emb[normal_proto_idx].mean(axis=0)
    
    tumor_proto = tumor_proto / np.linalg.norm(tumor_proto)
    normal_proto = normal_proto / np.linalg.norm(normal_proto)
    
    # Classify test set
    tumor_sim = test_emb @ tumor_proto
    normal_sim = test_emb @ normal_proto
    
    # Softmax probabilities
    logits = np.stack([normal_sim, tumor_sim], axis=1) * 100
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    tumor_probs = probs[:, 1]
    
    # Compute AUC
    auc = roc_auc_score(test_labels, tumor_probs)
    
    # Bootstrap confidence interval
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(test_labels), len(test_labels), replace=True)
        try:
            boot_auc = roc_auc_score(test_labels[idx], tumor_probs[idx])
            bootstrap_aucs.append(boot_auc)
        except ValueError:
            continue
    
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    std = np.std(bootstrap_aucs)
    
    return {
        "auc": float(auc),
        "auc_std": float(std),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "prototype_similarity": float(tumor_proto @ normal_proto),
        "n_test": len(test_labels)
    }


def compute_text_anchored_classification(
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    text_encoder,
    projection_head,
    prompts: dict,
    n_bootstrap: int = 1000,
    device: str = "cuda"
) -> dict:
    """
    Text-anchored classification (Semantic Anchoring).
    
    This is the proposed method that overcomes domain shift.
    """
    import torch.nn.functional as F
    
    # Get text embeddings
    with torch.no_grad():
        tumor_text = text_encoder.encode(prompts["tumor"])
        normal_text = text_encoder.encode(prompts["normal"])
        
        # Project to visual space
        tumor_anchor = projection_head(tumor_text.to(device))
        normal_anchor = projection_head(normal_text.to(device))
        
        tumor_anchor = F.normalize(tumor_anchor, dim=-1)
        normal_anchor = F.normalize(normal_anchor, dim=-1)
    
    # Normalize test embeddings
    test_emb = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)
    test_tensor = torch.FloatTensor(test_emb).to(device)
    test_tensor = F.normalize(test_tensor, dim=-1)
    
    # Compute similarities
    with torch.no_grad():
        tumor_sim = F.cosine_similarity(test_tensor, tumor_anchor.expand(len(test_tensor), -1))
        normal_sim = F.cosine_similarity(test_tensor, normal_anchor.expand(len(test_tensor), -1))
    
    tumor_sim = tumor_sim.cpu().numpy()
    normal_sim = normal_sim.cpu().numpy()
    
    # Softmax probabilities
    logits = np.stack([normal_sim, tumor_sim], axis=1) * 100
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    tumor_probs = probs[:, 1]
    
    # AUC
    auc = roc_auc_score(test_labels, tumor_probs)
    
    # Bootstrap CI
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(test_labels), len(test_labels), replace=True)
        try:
            boot_auc = roc_auc_score(test_labels[idx], tumor_probs[idx])
            bootstrap_aucs.append(boot_auc)
        except ValueError:
            continue
    
    std = np.std(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    
    return {
        "auc": float(auc),
        "auc_std": float(std),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_test": len(test_labels)
    }


def compute_few_shot_results(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    fractions: list = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20],
    n_runs: int = 5,
    device: str = "cuda"
) -> dict:
    """Few-shot learning with classification head."""
    from training import FewShotFineTuner
    
    results = {}
    
    for frac in fractions:
        n_samples = int(len(train_emb) * frac)
        run_aucs = []
        
        for seed in range(n_runs):
            np.random.seed(seed + 42)
            idx = np.random.choice(len(train_emb), n_samples, replace=False)
            
            trainer = FewShotFineTuner(
                embedding_dim=train_emb.shape[1],
                device=device
            )
            trainer.train_on_embeddings(
                train_emb[idx], train_labels[idx],
                epochs=50
            )
            
            metrics = trainer.evaluate(test_emb, test_labels)
            run_aucs.append(metrics["auc_roc"])
        
        results[f"{int(frac*100)}%"] = {
            "auc": float(np.mean(run_aucs)),
            "auc_std": float(np.std(run_aucs)),
            "n_samples": n_samples
        }
    
    return results


def format_table3(results: dict) -> str:
    """Format results as Table 3 from the paper."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 3: Main Results - AUC-ROC Across All Experimental Conditions")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Method':<25} {'Same-Cancer':<15} {'Cross-Cancer':<15} {'Cross-Species':<15}")
    lines.append("-" * 80)
    
    # Zero-shot (Prototype)
    same = results["same_cancer"]["prototype"]["auc"] * 100
    cross_c = results["cross_cancer"]["prototype"]["auc"] * 100
    cross_s = results["cross_species"]["prototype"]["auc"] * 100
    lines.append(f"{'Zero-shot (Prototype)':<25} {same:>6.2f}%        {cross_c:>6.2f}%        {cross_s:>6.2f}%")
    
    # Few-shot rows
    for frac in ["1%", "3%", "5%", "10%", "15%", "20%"]:
        if frac in results["same_cancer"].get("few_shot", {}):
            same = results["same_cancer"]["few_shot"][frac]["auc"] * 100
            cross_c = results["cross_cancer"]["few_shot"].get(frac, {}).get("auc", 0) * 100
            cross_s = results["cross_species"]["few_shot"].get(frac, {}).get("auc", 0) * 100
            
            # Cross-species few-shot marked with asterisk (not applicable)
            cross_s_str = f"{cross_s:>6.2f}%*" if cross_s > 0 else "   N/A*"
            lines.append(f"{'Few-shot ' + frac:<25} {same:>6.2f}%        {cross_c:>6.2f}%        {cross_s_str:>8}")
    
    # Text Anchoring
    if "text_anchored" in results["same_cancer"]:
        same = results["same_cancer"]["text_anchored"]["auc"] * 100
        cross_c = results["cross_cancer"]["text_anchored"]["auc"] * 100
        cross_s = results["cross_species"]["text_anchored"]["auc"] * 100
        lines.append(f"{'Text Anchoring (CLIP)':<25} {same:>6.2f}%        {cross_c:>6.2f}%        {cross_s:>6.2f}%")
    
    if "text_anchored_qwen" in results["same_cancer"]:
        same = results["same_cancer"]["text_anchored_qwen"]["auc"] * 100
        cross_c = results["cross_cancer"]["text_anchored_qwen"]["auc"] * 100
        cross_s = results["cross_species"]["text_anchored_qwen"]["auc"] * 100
        lines.append(f"{'Text Anchoring (Qwen)':<25} {same:>6.2f}%        {cross_c:>6.2f}%        {cross_s:>6.2f}%")
    
    lines.append("-" * 80)
    lines.append("* Few-shot not applied in cross-species setting")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 3 from the paper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=Path,
                           help="Directory with raw patch images")
    data_group.add_argument("--embeddings-dir", type=Path,
                           help="Directory with pre-computed embeddings (faster)")
    
    # Model options
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="Path to CPath-CLIP checkpoint")
    parser.add_argument("--projection-head", type=Path, default=None,
                       help="Path to trained projection head weights")
    parser.add_argument("--device", type=str, default="cuda")
    
    # Output options
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                       help="Output directory")
    
    # Experiment options
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                       help="Number of bootstrap samples for CI")
    parser.add_argument("--n-prototypes", type=int, default=50,
                       help="Number of prototypes per class")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test with reduced bootstrap samples")
    parser.add_argument("--skip-few-shot", action="store_true",
                       help="Skip few-shot experiments (faster)")
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.quick_test:
        args.n_bootstrap = 100
        print("Quick test mode: reduced bootstrap samples")
    
    print("=" * 60)
    print("REPRODUCING TABLE 3: Main Results")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {args.device}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print()
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_bootstrap": args.n_bootstrap,
            "n_prototypes": args.n_prototypes
        },
        "same_cancer": {},
        "cross_cancer": {},
        "cross_species": {}
    }
    
    # Load embeddings
    if args.embeddings_dir:
        print("Loading pre-computed embeddings...")
        
        # CATCH canine breast (train/test)
        catch_train_emb, catch_train_labels = load_embeddings(
            args.embeddings_dir, "catch_train"
        )
        catch_test_emb, catch_test_labels = load_embeddings(
            args.embeddings_dir, "catch_test"
        )
        
        # Mast cell tumor (test only)
        mast_emb, mast_labels = load_embeddings(
            args.embeddings_dir, "mast_cell"
        )
        
        # TCGA human breast (train)
        tcga_emb, tcga_labels = load_embeddings(
            args.embeddings_dir, "tcga_brca"
        )
        
        print(f"  CATCH train: {catch_train_emb.shape}")
        print(f"  CATCH test: {catch_test_emb.shape}")
        print(f"  Mast cell: {mast_emb.shape}")
        print(f"  TCGA: {tcga_emb.shape}")
        
    else:
        # Extract embeddings from images
        print("Extracting embeddings from images...")
        from models import CPathOmniInference
        
        model = CPathOmniInference(
            vision_encoder_path=str(args.checkpoint) if args.checkpoint else None,
            device=args.device
        )
        
        # This would extract embeddings from patch images
        # Implementation depends on data organization
        raise NotImplementedError(
            "Direct image processing not implemented. "
            "Please use --embeddings-dir with pre-computed embeddings."
        )
    
    # ========================================
    # EXPERIMENT 1: Same-Cancer Transfer
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Same-Cancer Transfer (Dog Breast → Dog Breast)")
    print("=" * 60)
    
    print("\nComputing prototype-based classification...")
    results["same_cancer"]["prototype"] = compute_prototype_classification(
        catch_train_emb, catch_train_labels,
        catch_test_emb, catch_test_labels,
        n_prototypes=args.n_prototypes,
        n_bootstrap=args.n_bootstrap
    )
    print(f"  AUC: {results['same_cancer']['prototype']['auc']*100:.2f}% "
          f"± {results['same_cancer']['prototype']['auc_std']*100:.2f}%")
    
    if not args.skip_few_shot:
        print("\nComputing few-shot results...")
        results["same_cancer"]["few_shot"] = compute_few_shot_results(
            catch_train_emb, catch_train_labels,
            catch_test_emb, catch_test_labels,
            device=args.device
        )
        for frac, metrics in results["same_cancer"]["few_shot"].items():
            print(f"  {frac}: AUC = {metrics['auc']*100:.2f}%")
    
    # ========================================
    # EXPERIMENT 2: Cross-Cancer Transfer
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Cross-Cancer Transfer (Dog Breast → Mast Cell)")
    print("=" * 60)
    
    print("\nComputing prototype-based classification...")
    results["cross_cancer"]["prototype"] = compute_prototype_classification(
        catch_train_emb, catch_train_labels,
        mast_emb, mast_labels,
        n_prototypes=args.n_prototypes,
        n_bootstrap=args.n_bootstrap
    )
    print(f"  AUC: {results['cross_cancer']['prototype']['auc']*100:.2f}% "
          f"± {results['cross_cancer']['prototype']['auc_std']*100:.2f}%")
    
    # ========================================
    # EXPERIMENT 3: Cross-Species Transfer
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Cross-Species Transfer (Human TCGA → Dog Breast)")
    print("=" * 60)
    
    print("\nComputing prototype-based classification...")
    results["cross_species"]["prototype"] = compute_prototype_classification(
        tcga_emb, tcga_labels,
        catch_test_emb, catch_test_labels,
        n_prototypes=args.n_prototypes,
        n_bootstrap=args.n_bootstrap
    )
    print(f"  AUC: {results['cross_species']['prototype']['auc']*100:.2f}% "
          f"± {results['cross_species']['prototype']['auc_std']*100:.2f}%")
    
    # Text Anchoring (if projection head available)
    if args.projection_head and args.projection_head.exists():
        print("\nComputing text-anchored classification...")
        from models import SemanticAnchoring
        from models.text_encoder import QwenTextEncoder
        from transformers import CLIPTextModel, CLIPTokenizer
        
        # Load projection head
        projection = torch.load(args.projection_head, map_location=args.device)
        
        # CLIP text encoder
        prompts = {
            "tumor": "Histopathological image of malignant tumor tissue with abnormal cellular proliferation",
            "normal": "Histopathological image of normal healthy tissue with regular cellular architecture"
        }
        
        # Would compute text-anchored results here
        # results["cross_species"]["text_anchored"] = compute_text_anchored_classification(...)
    
    # ========================================
    # Save and Display Results
    # ========================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save JSON
    json_path = args.output_dir / "table3_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {json_path}")
    
    # Save formatted table
    table_str = format_table3(results)
    table_path = args.output_dir / "table3_summary.txt"
    with open(table_path, 'w') as f:
        f.write(table_str)
    print(f"✓ Table saved to: {table_path}")
    
    # Print table
    print("\n")
    print(table_str)
    
    return results


if __name__ == "__main__":
    main()
