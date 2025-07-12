from pathlib import Path
from typing import Any, Dict, List
import re

import nibabel
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from jawfrac.data.datasets.base import VolumeDataset
import jawfrac.data.transforms as T

class MandibleSegDataset(VolumeDataset):
    """Dataset to load head scans with mandible segmentations."""

    def __init__(
        self,
        stage: str,
        root: str,
        regex_filter: str,
        regular_spacing: float,
        patch_size: int,
        stride: int,
        gamma_adjust: bool,
        max_patches_per_scan: int,
        ignore_outside: bool,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.regex_filter = regex_filter
        pre_transform = T.Compose(
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            T.PatchIndices(patch_size=patch_size, stride=stride),
            *((
                T.BonePatchIndices(),
                T.PositiveNegativeIndices(),
                T.MandibleStatistics(),
            ) if stage == 'fit' else ()),
        )

        super().__init__(
            stage=stage,
            root=root,
            pre_transform=pre_transform,
            **kwargs
        )

        print(f"Initializing MandibleSegDataset with root: {root}")
        print(f"Number of files found: {len(self.files)}")

    def _filter_files(self, files: List[Path]) -> List[Path]:
        if self.regex_filter:
            pattern = re.compile(self.regex_filter)
            return [f for f in files if pattern.search(str(f))]
        return files

    def load_inputs(
        self,
        file: Path,
    ) -> Dict[str, NDArray[Any]]:
        img = nibabel.load(self.root / file)
        intensities = np.asarray(img.dataobj)

        # convert 8-bit to 12-bit
        if intensities.min() == 0 and intensities.max() == 255:
            center = intensities[intensities > 0].mean()
            intensities = (intensities - center) / 255 * 4095

        # clip intensities to sensible range
        intensities = intensities.clip(-1024, 3096)

        return {
            'intensities': intensities.astype(np.int16),
            'spacing': np.array(img.header.get_zooms()),
            'orientation': nibabel.io_orientation(img.affine),
            'shape': np.array(img.header.get_data_shape()),
        }

    def load_target(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.bool8]]:
        seg = nibabel.load(self.root / file)
        labels = np.asarray(seg.dataobj)

        labels = ndimage.binary_closing(
            labels == 1, ndimage.generate_binary_structure(3, 3),
        )

        print(file)

        if file.stem == 'condyle_238.nii':
            k  = 3

        return {
            'labels': labels,
        }
