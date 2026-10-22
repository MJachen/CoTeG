import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BrainTumor3DDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: /home/cjc/brats2025
        """
        self.root_dir = root_dir
        self.transform = transform

        # 类别映射
        self.class_map = {
            "GLI": 0,
            "MEN": 1,
            "MET": 2
        }

        self.samples = self._scan_dataset()

    def _scan_dataset(self):
        samples = []

        for class_name in self.class_map:
            class_path = os.path.join(self.root_dir, class_name)

            for case in os.listdir(class_path):
                case_path = os.path.join(class_path, case)

                if os.path.isdir(case_path):
                    samples.append({
                        "case_path": case_path,
                        "label": self.class_map[class_name]
                    })

        print(f"共发现 {len(samples)} 个病例")
        return samples

    def _load_nifti(self, file_path):
        img = nib.load(file_path)
        data = img.get_fdata()
        return data.astype(np.float32)

    def _load_multimodal(self, case_path):
        """
        自动匹配 t1n / t1c / t2w / t2f
        """
        t1n = glob.glob(os.path.join(case_path, "*t1n*.nii*"))[0]
        t1c = glob.glob(os.path.join(case_path, "*t1c*.nii*"))[0]
        t2w = glob.glob(os.path.join(case_path, "*t2w*.nii*"))[0]
        t2f = glob.glob(os.path.join(case_path, "*t2f*.nii*"))[0]

        img_t1n = self._load_nifti(t1n)
        img_t1c = self._load_nifti(t1c)
        img_t2w = self._load_nifti(t2w)
        img_t2f = self._load_nifti(t2f)

        # 拼接为 4 通道
        image = np.stack([img_t1n, img_t1c, img_t2w, img_t2f], axis=0)

        return image

    def center_crop_or_pad_3d(self, x, target=(128, 128, 128)):
        # x: (C, H, W, D)
        C, H, W, D = x.shape
        th, tw, td = target

        out = np.zeros((C, th, tw, td), dtype=x.dtype)

        hs = max((H - th) // 2, 0);
        ws = max((W - tw) // 2, 0);
        ds = max((D - td) // 2, 0)
        he = hs + min(th, H);
        we = ws + min(tw, W);
        de = ds + min(td, D)

        ohs = max((th - H) // 2, 0);
        ows = max((tw - W) // 2, 0);
        ods = max((td - D) // 2, 0)
        ohe = ohs + (he - hs);
        owe = ows + (we - ws);
        ode = ods + (de - ds)

        out[:, ohs:ohe, ows:owe, ods:ode] = x[:, hs:he, ws:we, ds:de]
        return out

    def zscore_norm(self, x, eps=1e-6):
        # x: (C,H,W,D)
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        std = x.std(axis=(1, 2, 3), keepdims=True)
        return (x - mean) / (std + eps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        case_path = sample["case_path"]
        label = sample["label"]

        image = self._load_multimodal(case_path)
        image = self.zscore_norm(image)
        image = self.center_crop_or_pad_3d(image, (128, 128, 128))

        # 可选 transform
        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

