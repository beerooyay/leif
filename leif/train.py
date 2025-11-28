"""
training script for leif and baseline models

reproduces the paper results:
- leif-nano: perplexity 3.96
- baseline: perplexity 95.6
- 24x improvement
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .model import LeifModel, BaselineTransformer
from .data import LexiaDataset, build_vocabulary


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    is_leif: bool = True,
) -> torch.Tensor:
    """compute cross-entropy loss for next-token prediction"""
    tokens = batch["tokens"].to(device)
    
    if is_leif:
        senders = batch["senders"].to(device)
        receivers = batch["receivers"].to(device)
        conduits = batch["conduits"].to(device)
        positions = batch["positions"].to(device)
        
        logits = model(tokens, senders, receivers, conduits, positions)
    else:
        positions = batch["positions"].to(device)
        logits = model(tokens, positions)
    
    # shift for next-token prediction
    # predict token[i+1] from position i
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    
    # flatten
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=0,  # ignore padding
    )
    
    return loss


def compute_perplexity(loss: float) -> float:
    """convert cross-entropy loss to perplexity"""
    return math.exp(loss)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_leif: bool = True,
) -> float:
    """train for one epoch, return average loss"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch, device, is_leif)
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_leif: bool = True,
) -> Dict[str, float]:
    """evaluate model, return loss and perplexity"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss(model, batch, device, is_leif)
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = compute_perplexity(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


def train(
    model_type: str = "leif",
    data_path: str = "data/synthetic",
    output_path: str = "runs/leif",
    n_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 6e-4,
    max_seq_len: int = 128,
    device: str = "auto",
    seed: int = 42,
):
    """
    train a model and save results.
    
    args:
        model_type: "leif" or "baseline"
        data_path: path to dataset
        output_path: where to save model and logs
        n_epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        max_seq_len: maximum sequence length
        device: "auto", "cuda", "mps", or "cpu"
        seed: random seed for reproducibility
    """
    # set seed
    torch.manual_seed(seed)
    
    # device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    print(f"using device: {device}")
    
    # load data
    print(f"loading data from {data_path}...")
    dataset = LexiaDataset.load(data_path, max_seq_len)
    vocab_size = len(dataset.word2id)
    print(f"vocabulary size: {vocab_size}")
    print(f"total sequences: {len(dataset)}")
    
    # split: 70% train, 20% val, 10% test
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # create model
    is_leif = model_type == "leif"
    
    if is_leif:
        model = LeifModel(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=6,
            d_ff=1024,
            max_seq_len=max_seq_len,
            n_agents=16,
            n_conduits=4,
            dropout=0.1,
        )
    else:
        model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=6,
            d_ff=1024,
            max_seq_len=max_seq_len,
            dropout=0.1,
        )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model_type}, parameters: {n_params:,}")
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # training loop
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    history = []
    best_val_loss = float("inf")
    
    print(f"\ntraining for {n_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # train
        train_loss = train_epoch(model, train_loader, optimizer, device, is_leif)
        
        # evaluate
        val_metrics = evaluate(model, val_loader, device, is_leif)
        
        epoch_time = time.time() - start_time
        
        # log
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_perplexity": val_metrics["perplexity"],
            "time": epoch_time,
        })
        
        print(f"epoch {epoch+1:3d} | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_metrics['loss']:.4f} | "
              f"val ppl: {val_metrics['perplexity']:.2f} | "
              f"time: {epoch_time:.1f}s")
        
        # save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), output_path / "best_model.pt")
    
    print("-" * 60)
    
    # final test evaluation
    model.load_state_dict(torch.load(output_path / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, device, is_leif)
    
    print(f"\ntest results:")
    print(f"  loss: {test_metrics['loss']:.4f}")
    print(f"  perplexity: {test_metrics['perplexity']:.2f}")
    
    # compute attention density for leif
    if is_leif:
        sample_batch = next(iter(test_loader))
        senders = sample_batch["senders"]
        receivers = sample_batch["receivers"]
        density = model.get_attention_density(senders, receivers)
        print(f"  attention density: {density*100:.1f}%")
        test_metrics["attention_density"] = density
    
    # save results
    results = {
        "model_type": model_type,
        "n_params": n_params,
        "vocab_size": vocab_size,
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "history": history,
        "config": {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_len": max_seq_len,
            "seed": seed,
        }
    }
    
    if is_leif:
        results["attention_density"] = test_metrics.get("attention_density", None)
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nresults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train leif or baseline model")
    parser.add_argument("--model", type=str, default="leif", choices=["leif", "baseline"])
    parser.add_argument("--data", type=str, default="data/synthetic")
    parser.add_argument("--output", type=str, default="runs/leif")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train(
        model_type=args.model,
        data_path=args.data,
        output_path=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        device=args.device,
        seed=args.seed,
    )
