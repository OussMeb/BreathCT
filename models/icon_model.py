from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from utils.spatial import identity_dvf


class ICONInitializer(ABC):
    @abstractmethod
    def predict(self, moving: torch.Tensor, fixed: torch.Tensor, case_ids: list[str] | None = None) -> torch.Tensor:
        """Return initial pull DVF in canonical RAS CXYZ voxel convention."""


class IdentityInitializer(ICONInitializer):
    def predict(self, moving: torch.Tensor, fixed: torch.Tensor, case_ids: list[str] | None = None) -> torch.Tensor:
        return identity_dvf(
            batch=fixed.shape[0],
            shape_xyz=tuple(int(v) for v in fixed.shape[2:]),
            device=fixed.device,
            dtype=fixed.dtype,
        )


class PrecomputedDVFInitializer(ICONInitializer):
    """Loads precomputed canonical RAS CXYZ voxel DVFs.

    This is the safest bridge for uniGradICON:
    run uniGradICON separately, convert its output once, then train the refiner on fixed files.
    """

    def __init__(self, dvf_dir: str | Path):
        self.dvf_dir = Path(dvf_dir).expanduser()
        if not self.dvf_dir.exists():
            raise FileNotFoundError(f"DVF directory not found: {self.dvf_dir}")

    def predict(self, moving: torch.Tensor, fixed: torch.Tensor, case_ids: list[str] | None = None) -> torch.Tensor:
        if case_ids is None:
            raise ValueError("PrecomputedDVFInitializer requires case_ids.")

        fields: list[torch.Tensor] = []
        for case_id in case_ids:
            candidates = [
                self.dvf_dir / f"{case_id}_DVF_RAS_XYZC_voxel.nii.gz",
                self.dvf_dir / f"{case_id}_DVF_RAS_CXYZ_voxel.nii.gz",
                self.dvf_dir / f"{case_id}_DVF.nii.gz",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                searched = "\n".join(str(candidate) for candidate in candidates)
                raise FileNotFoundError(f"Missing precomputed DVF for {case_id}. Searched:\n{searched}")

            image = nib.load(str(path))
            data = np.asanyarray(image.dataobj).astype(np.float32)

            if data.shape[-1] == 3:
                data = np.moveaxis(data, -1, 0)
            if data.shape[0] != 3:
                raise ValueError(f"Expected DVF CXYZ or XYZC, got {data.shape}: {path}")

            field = torch.as_tensor(data, device=fixed.device, dtype=fixed.dtype)
            fields.append(field)

        dvf = torch.stack(fields, dim=0)
        if tuple(dvf.shape[2:]) != tuple(fixed.shape[2:]):
            raise ValueError(f"Precomputed DVF shape {dvf.shape} does not match fixed {fixed.shape}")

        return dvf


def build_initializer(kind: str = "identity", precomputed_dvf_dir: str | None = None) -> ICONInitializer:
    if kind == "identity":
        return IdentityInitializer()
    if kind == "precomputed":
        if precomputed_dvf_dir is None:
            raise ValueError("precomputed_dvf_dir is required for kind='precomputed'.")
        return PrecomputedDVFInitializer(precomputed_dvf_dir)
    raise ValueError(f"Unsupported initializer kind: {kind}")
