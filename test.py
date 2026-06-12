import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from options import TestOptions
from dataset import UniversalECGDataset
from model import UniversalECGResNet

def load_model_runtime(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = UniversalECGResNet(in_channels=ckpt["num_channels"], num_classes=len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["class_names"]

def main():
    opt = TestOptions().parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.test_output_dir, exist_ok=True)

    with open(os.path.join(opt.output_dir, "hierarchy_metadata.json"), "r") as f:
        meta = json.load(f)

    # 1. Initialize Parent Classifier
    parent_model, parent_classes = load_model_runtime(os.path.join(opt.model_dir, "best_main_parent_model.pt"), device)

    # 2. Initialize Available Downstream Sub-networks
    branch_models = {}
    for parent in meta["parent_classes"]:
        b_path = os.path.join(opt.model_dir, f"best_{parent}_branch.pt")
        if os.path.exists(b_path):
            model, sub_classes = load_model_runtime(b_path, device)
            branch_models[parent] = {"model": model, "classes": sub_classes}

    # 3. Load Manifest Tables Target Test Splits
    data_df = pd.read_csv(os.path.join(opt.output_dir, "main_multilabels.csv"))
    test_df = data_df[data_df["split"] == "test"].reset_index(drop=True)
    test_loader = DataLoader(UniversalECGDataset(test_df, label_cols=None, is_train=False), batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    with torch.no_grad():
        for signals, record_ids in tqdm(test_loader, desc="Hierarchical Inference"):
            signals = signals.to(device)
            parent_probs = torch.sigmoid(parent_model(signals)).cpu().numpy()

            for i in range(signals.size(0)):
                prob_row = parent_probs[i]
                top_parent_idx = np.argmax(prob_row)
                top_parent_cls = parent_classes[top_parent_idx]

                row_out = {
                    "unique_record_id": record_ids[i],
                    "predicted_parent_class": top_parent_cls,
                    "parent_probability": float(prob_row[top_parent_idx])
                }

                # Evaluate conditional pipeline against branch routing logic
                if top_parent_cls in branch_models:
                    b_engine = branch_models[top_parent_cls]["model"]
                    b_classes = branch_models[top_parent_cls]["classes"]
                    
                    branch_logits = b_engine(signals[i].unsqueeze(0))
                    branch_probs = torch.sigmoid(branch_logits).cpu().numpy()[0]
                    top_sub_idx = np.argmax(branch_probs)

                    row_out.update({
                        "predicted_subclass": b_classes[top_sub_idx],
                        "subclass_probability": float(branch_probs[top_sub_idx])
                    })
                else:
                    row_out.update({"predicted_subclass": "No Active Sub-Branch Model", "subclass_probability": np.nan})

                results.append(row_out)

    pd.DataFrame(results).to_csv(os.path.join(opt.test_output_dir, "hierarchical_predictions.csv"), index=False)
    print(f"Evaluation pipeline completely successful. Logs saved into: {opt.test_output_dir}")

if __name__ == "__main__":
    main()