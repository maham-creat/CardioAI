# Universal Hierarchical ECG Classification Engine

This repository provides a **dataset-agnostic, production-grade framework** for training and evaluating hierarchical multi-label deep learning models on Electrocardiogram (ECG) data. 

Unlike rigid architectures tied to a single dataset, this pipeline automatically discovers category relationships, handles any arbitrary channel configuration (e.g., 12-lead, 3-lead, or single-lead formats), and routes classification logic down an inferred tree structure: **Global Parent Classes → Isolated Subclass Expert Branches.**

---

## 📂 Repository Architecture
```text

universal_ecg_project/
│
├── README.md               # Project documentation and quick-start guide
├── options.py              # Central runtime argument parser configurations
├── dataset.py              # Generic, standardized NumPy matrix loader 
├── model.py                # Adaptable ResNet-18 spatial 1D/2D processor
│
├── prepare_data.py         # Data standardizer (Translates raw formats to .npy structures)
├── train.py                # Core hierarchical model optimizer orchestrator
└── test.py                 # Sequential conditional routing inference loop

+-------------------+      +---------------------+      +---------------------+
|  prepare_data.py  | ---> |      train.py       | ---> |       test.py       |
+-------------------+      +---------------------+      +---------------------+
| Loads custom data |      | Trains Parent Model |      | Runs Parent Network |
| Saves standardized|      | Slices target logs  |      | Selects top class   |
| NumPy arrays and  |      | Trains individual   |      | Conditionally routes|
| a hierarchy JSON  |      | subclass branches   |      | to Subclass Experts |
+-------------------+      +---------------------+      +---------------------+


## Setup & Environment
pip install torch torchvision numpy pandas tqdm scikit-learn
## Prepare Your Custom Dataset
python prepare_data.py --raw_data_path "./data/my_raw_dataset" --output_dir "./processed_dataset"

## Run Hierarchical Training
python train.py --output_dir "./processed_dataset" --model_dir "./saved_models" --batch_size 128 --epochs 35 --lr 5e-4

##TEST
python test.py --output_dir "./processed_dataset" --model_dir "./saved_models" --test_output_dir "./eval_results" --threshold 0.50
