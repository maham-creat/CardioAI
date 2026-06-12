# config.py
import os

# The only local path needed is where your trained PyTorch models live
MODEL_DIR = "./hierarchical_resnet_models"

THRESHOLD_MAIN = 0.5

CLINICAL_MAPPING = {
    "NORM": {"name": "Normal Function", "desc": "Your ECG shows typical electrical heart activity within normal parameters.", "severity": "normal"},
    "MI": {"name": "Potential Muscle Injury", "desc": "Indicators suggest potential strain or historical tissue injury in the heart muscle.", "severity": "critical"},
    "STTC": {"name": "ST/T Wave Variations", "desc": "Minor structural or metabolic variants detected in the recovery phase of the heartbeat.", "severity": "warning"},
    "CD": {"name": "Conduction Variation", "desc": "A slight shift in how electrical signals travel through your heart's chambers.", "severity": "warning"},
    "HYP": {"name": "Muscle Thickening Indicators", "desc": "Features point toward possible thickening (hypertrophy) of the heart walls.", "severity": "warning"}
}