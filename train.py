import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from options import TrainOptions
from dataset import UniversalECGDataset
from model import UniversalECGResNet

def train_network(csv_path, class_names, name, epochs, opt, num_channels, device):
    print("\n" + "="*80)
    print(f"TRAINING MODULE: {name} | Targets: {class_names}")
    print("="*80)

    data = pd.read_csv(csv_path)
    train_ds = UniversalECGDataset(data[data["split"] == "train"], class_names, is_train=True)
    val_ds = UniversalECGDataset(data[data["split"] == "val"], class_names, is_train=True)
    
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    model = UniversalECGResNet(in_channels=num_channels, num_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    
    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                logits = model(signals.to(device))
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())
                
        val_f1 = f1_score(np.vstack(all_labels), (np.vstack(all_probs) >= 0.5).astype(int), average="macro", zero_division=0)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_channels": num_channels
            }, os.path.join(opt.model_dir, f"best_{name}.pt"))
            print(f" -> Checkpoint Optimized [Val F1: {val_f1:.4f}]")

def main():
    opt = TrainOptions().parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    os.makedirs(opt.model_dir, exist_ok=True)

    # Read configuration mappings discovered by preprocessing
    with open(os.path.join(opt.output_dir, "hierarchy_metadata.json"), "r") as f:
        meta = json.load(f)

    # 1. Train Global Top Level Network Module
    main_csv = os.path.join(opt.output_dir, "main_multilabels.csv")
    train_network(main_csv, meta["parent_classes"], "main_parent_model", opt.epochs, opt, meta["num_channels"], device)

    # 2. Iterate and Train Sub-level Branch Architectures Automatically
    for parent_cls, sub_classes in meta["branches"].items():
        branch_csv = os.path.join(opt.output_dir, "branches", parent_cls, f"{parent_cls}_branch_multilabels.csv")
        train_network(branch_csv, sub_classes, f"{parent_cls}_branch", opt.epochs, opt, meta["num_channels"], device)

if __name__ == "__main__":
    main()