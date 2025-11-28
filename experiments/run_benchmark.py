"""
simple professional benchmark - runs leif vs baseline and generates plots
"""

import json
import subprocess
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_training(model_type: str, data_path: str, output_path: str, epochs: int, batch_size: int):
    """run training via subprocess to avoid MPS issues"""
    cmd = [
        "python3", "-m", "leif.train",
        "--model", model_type,
        "--data", data_path,
        "--output", output_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    
    print(f"\n{'='*60}")
    print(f"🧪 training {model_type}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/beerooyay/Desktop/leif")
    elapsed = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # load results
    results_path = Path(output_path) / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        results["train_time"] = elapsed
        return results
    else:
        print(f"❌ training failed for {model_type}")
        return None


def create_plots(results: dict, output_dir: str):
    """create professional visualizations"""
    
    print(f"\n📈 creating professional plots...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    models = list(results.keys())
    perplexities = [results[m]["test_perplexity"] for m in models]
    densities = [results[m].get("attention_density", 1.0) or 1.0 for m in models]
    train_times = [results[m]["train_time"]/60 for m in models]
    n_params = [results[m]["n_params"]/1e6 for m in models]
    
    # color scheme
    colors = ['#e74c3c' if 'baseline' in m else '#3498db' for m in models]
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("LEIF vs Baseline: Professional Benchmark\n10K Multi-Party Conversations (5-8 Agents)", 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # 1. perplexity
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(models, perplexities, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Test Perplexity ↓", fontsize=12, fontweight='bold')
    ax1.set_title("Prediction Accuracy", fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars1, perplexities):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. attention density
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(models, [d*100 for d in densities], color=colors, alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel("Attention Density % ↓", fontsize=12, fontweight='bold')
    ax2.set_title("Computational Efficiency", fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Full Attention')
    for bar, val in zip(bars2, densities):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. compute savings
    ax3 = plt.subplot(2, 3, 3)
    savings = [(1 - d) * 100 for d in densities]
    bars3 = ax3.bar(models, savings, color=['#27ae60' if s > 0 else '#95a5a6' for s in savings], 
                    alpha=0.85, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel("Compute Savings % ↑", fontsize=12, fontweight='bold')
    ax3.set_title("Efficiency Gains", fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars3, savings):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 4. training time
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(models, train_times, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel("Training Time (min)", fontsize=12, fontweight='bold')
    ax4.set_title("Training Speed", fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars4, train_times):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val:.1f}m', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 5. 3D scatter - perplexity vs density vs params
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    x = np.array([d*100 for d in densities])
    y = np.array(n_params)
    z = np.array(perplexities)
    
    for i, (xi, yi, zi, model, color) in enumerate(zip(x, y, z, models, colors)):
        ax5.scatter([xi], [yi], [zi], s=200, c=[color], alpha=0.9, edgecolors='black', linewidth=2)
        ax5.text(xi, yi, zi + 0.05, model, fontsize=9, fontweight='bold', ha='center')
    
    ax5.set_xlabel("Attention %", fontsize=10, fontweight='bold')
    ax5.set_ylabel("Params (M)", fontsize=10, fontweight='bold')
    ax5.set_zlabel("Perplexity", fontsize=10, fontweight='bold')
    ax5.set_title("3D Performance Space", fontsize=14, fontweight='bold')
    
    # 6. training curves
    ax6 = plt.subplot(2, 3, 6)
    for model, color in zip(models, colors):
        if "history" in results[model]:
            history = results[model]["history"]
            epochs_range = range(1, len(history["val_loss"]) + 1)
            ax6.plot(epochs_range, history["val_loss"], color=color, linewidth=2.5, 
                    label=f'{model}', marker='o', markersize=4)
    
    ax6.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax6.set_ylabel("Validation Loss", fontsize=12, fontweight='bold')
    ax6.set_title("Training Dynamics", fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10, loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / "benchmark_results.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_path / "benchmark_results.pdf", bbox_inches='tight')
    
    print(f"📊 saved plots to {output_dir}")
    
    return fig


def create_heatmap(results: dict, output_dir: str):
    """create attention density heatmap visualization"""
    
    print(f"\n🔥 creating attention heatmap...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    metrics = ['Perplexity', 'Attention %', 'Compute Savings %', 'Train Time (min)']
    
    data = []
    for m in models:
        r = results[m]
        density = r.get("attention_density", 1.0) or 1.0
        data.append([
            r["test_perplexity"],
            density * 100,
            (1 - density) * 100,
            r["train_time"] / 60
        ])
    
    data = np.array(data)
    
    # normalize each column for heatmap
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
    
    sns.heatmap(data_norm, annot=data, fmt='.2f', cmap='RdYlGn_r',
                xticklabels=metrics, yticklabels=models, ax=ax,
                linewidths=2, linecolor='white', cbar_kws={'label': 'Normalized Score'})
    
    ax.set_title("Performance Heatmap\n(Lower perplexity & attention = better)", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "benchmark_heatmap.png", dpi=300, bbox_inches='tight',
                facecolor='white')
    
    print(f"🔥 saved heatmap to {output_dir}")


def create_summary(results: dict, output_dir: str):
    """create summary table"""
    
    baseline_ppl = results.get("baseline", {}).get("test_perplexity", 1)
    
    data = []
    for model, r in results.items():
        density = r.get("attention_density", 1.0) or 1.0
        ppl_improvement = ((baseline_ppl / r["test_perplexity"]) - 1) * 100 if model != "baseline" else 0
        
        data.append({
            "Model": model,
            "Perplexity": f"{r['test_perplexity']:.3f}",
            "Attention %": f"{density*100:.1f}%",
            "Compute Savings": f"{(1-density)*100:.1f}%",
            "PPL Improvement": f"+{ppl_improvement:.1f}%" if ppl_improvement > 0 else f"{ppl_improvement:.1f}%",
            "Parameters": f"{r['n_params']/1e6:.2f}M",
            "Train Time": f"{r['train_time']/60:.1f} min",
        })
    
    df = pd.DataFrame(data)
    df.to_csv(Path(output_dir) / "benchmark_summary.csv", index=False)
    
    print(f"\n📋 BENCHMARK SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # save as markdown too
    with open(Path(output_dir) / "benchmark_summary.md", "w") as f:
        f.write("# LEIF Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Key Findings\n\n")
        
        leif_results = {k: v for k, v in results.items() if "leif" in k.lower()}
        if leif_results:
            best_leif = min(leif_results.items(), key=lambda x: x[1]["test_perplexity"])
            baseline = results.get("baseline", {})
            
            if baseline:
                ppl_gain = ((baseline["test_perplexity"] / best_leif[1]["test_perplexity"]) - 1) * 100
                compute_save = (1 - (best_leif[1].get("attention_density", 1) or 1)) * 100
                
                f.write(f"- **Best LEIF model**: {best_leif[0]}\n")
                f.write(f"- **Perplexity improvement**: {ppl_gain:.1f}% better than baseline\n")
                f.write(f"- **Compute savings**: {compute_save:.1f}% less attention computation\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/multiparty_large")
    parser.add_argument("--output", type=str, default="benchmark_results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # run baseline
    r = run_training("baseline", args.data, str(output_dir / "baseline"), args.epochs, args.batch_size)
    if r:
        results["baseline"] = r
    
    # run leif
    r = run_training("leif", args.data, str(output_dir / "leif"), args.epochs, args.batch_size)
    if r:
        results["leif"] = r
    
    # save combined results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # create visualizations
    if len(results) >= 2:
        create_plots(results, str(output_dir))
        create_heatmap(results, str(output_dir))
        create_summary(results, str(output_dir))
        
        print(f"\n🎉 BENCHMARK COMPLETE!")
        print(f"📁 Results saved to {output_dir}/")
    else:
        print("❌ Not enough results to create plots")
