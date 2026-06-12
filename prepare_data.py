import os
import shutil
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from options import BaseOptions

def main():
    opt = BaseOptions().parse()
    np.random.seed(opt.seed)
    
    if os.path.exists(opt.output_dir):
        shutil.rmtree(opt.output_dir)
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "extracted_signals"), exist_ok=True)

    # -------------------------------------------------------------------------
    # TODO: PLUG IN YOUR PARSING LOGIC HERE
    # Generate a DataFrame containing these exact structural columns:
    # ['unique_record_id', 'split', 'standardized_signal_path', 'parent_classes', 'sub_classes']
    # 'parent_classes' & 'sub_classes' should be clean Python lists of strings.
    # -------------------------------------------------------------------------
    # (Mocking 12-channel ECG data with a dynamic hierarchy for demonstration)
    mock_data = []
    for idx in range(600):
        p_cls = "PARENT_A" if idx % 2 == 0 else "PARENT_B"
        s_cls = f"{p_cls}_SUB_{idx % 3}"
        sig_path = os.path.join(opt.output_dir, "extracted_signals", f"REC_{idx:05d}.npy")
        np.save(sig_path, np.random.randn(12, 1000).astype(np.float32))
        
        mock_data.append({
            "unique_record_id": f"REC_{idx:05d}",
            "split": "train" if idx < 400 else ("val" if idx < 500 else "test"),
            "standardized_signal_path": sig_path,
            "parent_classes": [p_cls],
            "sub_classes": [s_cls]
        })
    df = pd.DataFrame(mock_data)
    # -------------------------------------------------------------------------

    # 1. Process and save Global Parent Labels Matrix
    all_parents = sorted(list(set([p for row in df["parent_classes"] for p in row])))
    parent_mlb = MultiLabelBinarizer(classes=all_parents)
    parent_encoded = pd.DataFrame(parent_mlb.fit_transform(df["parent_classes"]), columns=all_parents)
    
    main_manifest = pd.concat([df[["unique_record_id", "split", "standardized_signal_path"]], parent_encoded], axis=1)
    main_manifest.to_csv(os.path.join(opt.output_dir, "main_multilabels.csv"), index=False)

    # 2. Process and save Branch Subclass Models dynamically
    branch_summary = {}
    for parent in all_parents:
        # Filter rows where this parent class is present
        branch_df = df[df["parent_classes"].apply(lambda x: parent in x)].copy()
        
        # Determine unique subclasses under this specific parent family tree
        subclasses_in_branch = sorted(list(set([s for row in branch_df["sub_classes"] for s in row])))
        
        if len(subclasses_in_branch) >= 2:
            branch_dir = os.path.join(opt.output_dir, "branches", parent)
            os.makedirs(branch_dir, exist_ok=True)
            
            branch_mlb = MultiLabelBinarizer(classes=subclasses_in_branch)
            branch_encoded = pd.DataFrame(branch_mlb.fit_transform(branch_df["sub_classes"]), columns=subclasses_in_branch)
            
            branch_manifest = pd.concat([branch_df[["unique_record_id", "split", "standardized_signal_path"]].reset_index(drop=True), branch_encoded], axis=1)
            branch_manifest.to_csv(os.path.join(branch_dir, f"{parent}_branch_multilabels.csv"), index=False)
            
            # Save the structural keys for the training loop
            pd.Series(subclasses_in_branch).to_csv(os.path.join(branch_dir, f"{parent}_kept_codes.csv"), index=False, header=["code"])
            branch_summary[parent] = subclasses_in_branch

    # 3. Save comprehensive structural config artifact
    num_channels = int(np.load(df.iloc[0]["standardized_signal_path"]).shape[0])
    with open(os.path.join(opt.output_dir, "hierarchy_metadata.json"), "w") as f:
        json.dump({
            "parent_classes": all_parents,
            "num_channels": num_channels,
            "branches": branch_summary
        }, f, indent=4)

    print(f"Extraction Completed. Inferred Architecture: {list(branch_summary.keys())}")

if __name__ == "__main__":
    main()