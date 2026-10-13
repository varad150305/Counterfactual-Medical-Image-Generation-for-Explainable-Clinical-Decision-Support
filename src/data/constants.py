"""CheXpert label schema, verified against PRISM's finetune_chexpert.py
(docs/PHASE_1_AUDIT.md section D) and the standard CheXpert release."""

# All 14 CheXpert pathology columns, in the dataset's own column order.
CHEXPERT_ALL_FINDINGS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# This project's target subset (CLAUDE.md section 5).
DEFAULT_TARGET_FINDINGS = [
    "No Finding",
    "Pneumonia",
    "Pleural Effusion",
    "Cardiomegaly",
]

# CheXpert label values: 1 = positive, 0 = negative, -1 = uncertain, NaN = unmentioned.
POSITIVE = 1
NEGATIVE = 0
UNCERTAIN = -1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
