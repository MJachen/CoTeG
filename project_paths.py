import glob
import os


def get_pack_dir(default="./brats2021_hgg_lgg_packs"):
    return os.environ.get("PACK_DIR", default)


def find_xfull_path(pack_dir):
    env_path = os.environ.get("X_FULL_PATH")
    if env_path:
        return env_path

    preferred = [
        os.path.join(pack_dir, "X_full_4mod.pt"),
        os.path.join(pack_dir, "X_full_4mod_gli_met.pt"),
        os.path.join(os.path.dirname(os.path.abspath(pack_dir)), "X_full_4mod.pt"),
        os.path.join(os.path.dirname(os.path.abspath(pack_dir)), "X_full_4mod_gli_met.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    hits = sorted(glob.glob(os.path.join(pack_dir, "X_full_4mod*.pt")))
    if hits:
        return hits[0]
    parent_hits = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(pack_dir)), "X_full_4mod*.pt")))
    if parent_hits:
        return parent_hits[0]

    return preferred[0]


def find_train_pack_path(pack_dir):
    env_path = os.environ.get("TRAIN_AUG_PATH")
    if env_path:
        return env_path

    preferred = [
        os.path.join(pack_dir, "train_aug_15mods.pt"),
        os.path.join(pack_dir, "train_aug_k1to4.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path
    return preferred[0]


def get_weight_path(env_name, local_name, legacy_path=None):
    env_path = os.environ.get(env_name)
    if env_path:
        return env_path
    if os.path.exists(local_name):
        return local_name
    return legacy_path or local_name
