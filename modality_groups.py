MODALITY_ORDER = ["T1", "T1CE", "T2", "FLAIR"]

MODALITY_FILE_KEYS = {
    "T1": "t1",
    "T1CE": "t1ce",
    "T2": "t2",
    "FLAIR": "flair",
}

MODALITY_GROUPS = [
    ("full", ["T1", "T1CE", "T2", "FLAIR"]),
    ("t1", ["T1"]),
    ("t1ce", ["T1CE"]),
    ("t2", ["T2"]),
    ("flair", ["FLAIR"]),
    ("t1_t1ce", ["T1", "T1CE"]),
    ("t1_t2", ["T1", "T2"]),
    ("t1_flair", ["T1", "FLAIR"]),
    ("t1ce_t2", ["T1CE", "T2"]),
    ("t1ce_flair", ["T1CE", "FLAIR"]),
    ("t2_flair", ["T2", "FLAIR"]),
    ("t1_t1ce_t2", ["T1", "T1CE", "T2"]),
    ("t1_t1ce_flair", ["T1", "T1CE", "FLAIR"]),
    ("t1_t2_flair", ["T1", "T2", "FLAIR"]),
    ("t1ce_t2_flair", ["T1CE", "T2", "FLAIR"]),
]


def group_names():
    return [name for name, _ in MODALITY_GROUPS]


def group_to_omega_row(present_modalities):
    present = set(present_modalities)
    return [1.0 if mod in present else 0.0 for mod in MODALITY_ORDER]


def modalities_to_file_keys(modalities=None):
    modalities = MODALITY_ORDER if modalities is None else modalities
    return [MODALITY_FILE_KEYS[m] for m in modalities]
