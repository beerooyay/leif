"""
professional benchmark suite for leif vs baseline
comprehensive testing across different conversation complexities
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

import torch
from leif.train import train, evaluate
from leif.data import LexiaDataset
from leif.model import LeifModel, BaselineTransformer


def run_comprehensive_benchmark(
    data_path: str,
    output_dir: str,
    model_configs: List[Dict],
    epochs: int = 20,
    batch_size: int = 32,
) -> Dict:
    """run full benchmark suite across multiple model configurations"""
    
    print(f"🚀 starting professional benchmark on {data_path}")
    print(f"📊 testing {len(model_configs)} configurations")
    
    # load data once
    dataset = LexiaDataset.load(data_path)
    vocab_size = len(dataset.word2id)
    n_agents = max([conv.n_agents for conv in dataset.conversations])
    
    results = {}
    
    for i, config in enumerate(model_configs):
        model_name = config["name"]
        print(f"\n{'='*60}")
        print(f"🧪 testing {model_name} ({i+1}/{len(model_configs)})")
        print(f"{'='*60}")
        
        # create model
        if config["type"] == "leif":
            model = LeifModel(
                vocab_size=vocab_size,
                n_agents=n_agents,
                n_conduits=1,
                max_seq_len=config["max_seq_len"],
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                d_ff=config["d_ff"],
                dropout=0.1,
                mask_config=config.get("mask_config"),
            )
        else:
            model = BaselineTransformer(
                vocab_size=vocab_size,
                max_seq_len=config["max_seq_len"],
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                d_ff=config["d_ff"],
                dropout=0.1,
            )
        
        # count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"📏 {n_params:,} parameters")
        
        # train
        start_time = time.time()
        history = train(
            model_type=config["type"],
            data_path=data_path,
            output_path=Path(output_dir) / model_name,
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=config.get("lr", 1e-4),
            max_seq_len=config["max_seq_len"],
            device="mps",  # m4 pro max
        )
        train_time = time.time() - start_time
        
        # evaluate
        model.load_state_dict(torch.load(Path(output_dir) / model_name / "best_model.pt"))
        
        # get test set
        from torch.utils.data import random_split
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.2 * n_total)
        n_test = n_total - n_train - n_val
        
        _, _, test_dataset = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        test_metrics = evaluate(model, test_loader, torch.device("mps"), config["type"] == "leif")
        
        # compute attention density for leif
        attention_density = None
        if config["type"] == "leif":
            sample_batch = next(iter(test_loader))
            senders = sample_batch["senders"]
            receivers = sample_batch["receivers"]
            attention_density = model.get_attention_density(senders, receivers)
        
        results[model_name] = {
            "config": config,
            "n_params": n_params,
            "train_time": train_time,
            "test_loss": test_metrics["loss"],
            "test_perplexity": test_metrics["perplexity"],
            "attention_density": attention_density,
            "history": history,
        }
        
        print(f"✅ {model_name} complete")
        print(f"   perplexity: {test_metrics['perplexity']:.3f}")
        if attention_density:
            print(f"   attention density: {attention_density*100:.1f}%")
        print(f"   train time: {train_time/60:.1f} min")
    
    # save results
    with open(Path(output_dir) / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def create_professional_plots(results: Dict, output_dir: str):
    """create publication-quality visualizations"""
    
    print(f"\n📈 creating professional plots...")
    
    # set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # extract data for plotting
    models = list(results.keys())
    perplexities = [results[m]["test_perplexity"] for m in models]
    attention_densities = [results[m].get("attention_density", 1.0) for m in models]
    train_times = [results[m]["train_time"]/60 for m in models]  # minutes
    n_params = [results[m]["n_params"]/1e6 for m in models]  # millions
    
    # create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Leif vs Baseline: Professional Benchmark Results", fontsize=20, fontweight='bold')
    
    # 1. perplexity comparison
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(models, perplexities, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel("Test Perplexity", fontsize=12, fontweight='bold')
    ax1.set_title("Prediction Accuracy", fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # add value labels on bars
    for bar, val in zip(bars1, perplexities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. attention density (efficiency)
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(models, [d*100 for d in attention_densities], 
                    alpha=0.8, edgecolor='black', linewidth=1, color='orange')
    ax2.set_ylabel("Attention Density (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Computational Efficiency", fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    
    # add value labels
    for bar, val in zip(bars2, attention_densities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. training time
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(models, train_times, alpha=0.8, edgecolor='black', 
                    linewidth=1, color='green')
    ax3.set_ylabel("Training Time (minutes)", fontsize=12, fontweight='bold')
    ax3.set_title("Training Speed", fontsize=14)
    ax3.tick_params(axis='x', rotation=45)
    
    # add value labels
    for bar, val in zip(bars3, train_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    # 4. parameter count
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(models, n_params, alpha=0.8, edgecolor='black', 
                    linewidth=1, color='red')
    ax4.set_ylabel("Parameters (millions)", fontsize=12, fontweight='bold')
    ax4.set_title("Model Size", fontsize=14)
    ax4.tick_params(axis='x', rotation=45)
    
    # add value labels
    for bar, val in zip(bars4, n_params):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}M', ha='center', va='bottom', fontweight='bold')
    
    # 5. 3D heatmap: perplexity vs attention density vs parameters
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    
    # create mesh for surface plot
    x = np.array([d*100 for d in attention_densities])  # attention density %
    y = np.array(n_params)  # parameters
    z = np.array(perplexities)  # perplexity
    
    # scatter plot with size based on training time
    scatter = ax5.scatter(x, y, z, s=[t*10 for t in train_times], 
                         c=range(len(models)), cmap='viridis', 
                         alpha=0.8, edgecolors='black', linewidth=2)
    
    ax5.set_xlabel("Attention Density (%)", fontsize=10, fontweight='bold')
    ax5.set_ylabel("Parameters (M)", fontsize=10, fontweight='bold')
    ax5.set_zlabel("Perplexity", fontsize=10, fontweight='bold')
    ax5.set_title("3D Performance Landscape", fontsize=14)
    
    # add model labels
    for i, (xi, yi, zi, model) in enumerate(zip(x, y, z, models)):
        ax5.text(xi, yi, zi, model, fontsize=9, fontweight='bold')
    
    # 6. training curves over time
    ax6 = plt.subplot(2, 3, 6)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    for i, (model, color) in enumerate(zip(models, colors)):
        history = results[model]["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax6.plot(epochs, history["train_loss"], color=color, alpha=0.7, 
                linestyle='--', label=f'{model} (train)')
        ax6.plot(epochs, history["val_loss"], color=color, alpha=0.9, 
                linewidth=2, label=f'{model} (val)')
    
    ax6.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax6.set_ylabel("Loss", fontsize=12, fontweight='bold')
    ax6.set_title("Training Dynamics", fontsize=14)
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save high-quality version
    plt.savefig(Path(output_dir) / "benchmark_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / "benchmark_results.pdf", bbox_inches='tight')
    
    print(f"📊 plots saved to {output_dir}")
    
    return plt


def create_summary_table(results: Dict, output_dir: str):
    """create a clean summary table"""
    
    data = []
    for model_name, result in results.items():
        data.append({
            "Model": model_name,
            "Type": result["config"]["type"],
            "Params (M)": f"{result['n_params']/1e6:.2f}",
            "Test PPL": f"{result['test_perplexity']:.3f}",
            "Attention %": f"{result.get('attention_density', 1.0)*100:.1f}%",
            "Train Time (m)": f"{result['train_time']/60:.1f}",
            "PPL Improvement": f"{(results['baseline']['test_perplexity'] / result['test_perplexity'] - 1)*100:.1f}%" if model_name != 'baseline' else "baseline",
            "Compute Savings": f"{(1 - result.get('attention_density', 1.0))*100:.1f}%" if result.get('attention_density') else "N/A",
        })
    
    df = pd.DataFrame(data)
    
    # save as csv and styled html
    df.to_csv(Path(output_dir) / "benchmark_summary.csv", index=False)
    
    # create html table with styling
    html = df.to_html(index=False, classes='table table-striped', 
                      table_id='benchmark-table')
    
    with open(Path(output_dir) / "benchmark_summary.html", "w") as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Leif Benchmark Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .table {{ font-size: 14px; }}
        .table th {{ background-color: #f8f9fa; font-weight: bold; }}
        .metric-positive {{ color: #28a745; font-weight: bold; }}
        .metric-neutral {{ color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Leif Professional Benchmark Results</h1>
        {html}
    </div>
</body>
</html>
""")
    
    print(f"📋 summary table saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="run professional benchmark")
    parser.add_argument("--data", type=str, required=True, help="path to dataset")
    parser.add_argument("--output", type=str, default="benchmark_results", help="output directory")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    
    args = parser.parse_args()
    
    # model configurations to test
    model_configs = [
        {
            "name": "baseline",
            "type": "baseline",
            "max_seq_len": 128,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1024,
            "lr": 1e-4,
        },
        {
            "name": "leif_default",
            "type": "leif",
            "max_seq_len": 128,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1024,
            "lr": 1e-4,
        },
        {
            "name": "leif_sparse",
            "type": "leif",
            "max_seq_len": 128,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1024,
            "lr": 1e-4,
            "mask_config": {
                "same_sender": True,
                "direct_address": True,
                "temporal_window": 4,  # more restrictive
                "include_self": False,
            }
        },
        {
            "name": "leif_large",
            "type": "leif",
            "max_seq_len": 128,
            "d_model": 512,
            "n_heads": 16,
            "n_layers": 8,
            "d_ff": 2048,
            "lr": 5e-5,
        }
    ]
    
    # create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # run benchmark
    results = run_comprehensive_benchmark(
        data_path=args.data,
        output_dir=str(output_dir),
        model_configs=model_configs,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    # create visualizations
    create_professional_plots(results, str(output_dir))
    create_summary_table(results, str(output_dir))
    
    print(f"\n🎉 professional benchmark complete!")
    print(f"📁 results saved to {output_dir}")
    print(f"📊 open {output_dir}/benchmark_summary.html for interactive results")
