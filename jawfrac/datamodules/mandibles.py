from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import os
from torchtyping import TensorType

from jawfrac.data.datasets.mandibles import MandibleSegDataset
import jawfrac.data.transforms as T
from jawfrac.datamodules.base import VolumeDataModule
from jawfrac.datamodules.jawfrac import JawFracDataModule


class MandibleSegDataModule(VolumeDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        patch_size: int,
        gamma_adjust: bool,
        max_patches_per_scan: int,
        ignore_outside: bool,
        regular_spacing: List[float],
        stride: List[int],
        regex_filter: str = '',
        exclude: List[str] = [],
        val_size: float = 0.2,
        test_size: float = 0.1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(
            root=root,
            exclude=exclude,
            regex_filter=regex_filter,  # Add this line
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            seed=seed,
        )
        self.patch_size = patch_size
        self.gamma_adjust = gamma_adjust
        self.max_patches_per_scan = max_patches_per_scan
        self.ignore_outside = ignore_outside
        self.regular_spacing = regular_spacing
        self.stride = stride
        self.regex_filter = regex_filter

    def _filter_files(self, pattern: str) -> List[Path]:
        files = super()._filter_files(pattern)
        
        overview_file = self.root / 'Fabian overview.csv'
        if not overview_file.exists():
            return files

        df = pd.read_csv(overview_file)
        df = df[pd.isna(df['Note']) & ~pd.isna(df['Complete'])]
        df = df[~df['Complete'].str.match(r'.*[,+]')]
        pseudonyms = df['Pseudonym'].tolist()

        dirs = list(map(lambda p: p.parent.stem, files))
        files = [f for f, d in zip(files, dirs) if d in pseudonyms]

        return files

    def _files(self, stage: str) -> List[Tuple[Path, ...]]:
        scan_files = self._filter_files('**/*.nii.gz')

        if stage == 'predict':
            return list(zip(scan_files))

        seg_files = self._filter_files('**/*_*.nii.gz')
        seg_files = sorted(set(seg_files) - set(scan_files))

        return list(zip(scan_files, seg_files))

    '''
    def setup(self, stage: Optional[str] = None) -> None:
        print(f"Setting up MandibleSegDataModule with root: {self.root}")
        print(f"Files in root directory: {os.listdir(self.root)}")

        self.train_dataset = MandibleSegDataset(
            stage='fit',
            root=self.root,
            regex_filter=self.regex_filter,
            regular_spacing=self.regular_spacing[0],
            patch_size=self.patch_size,
            stride=self.stride[0],
            gamma_adjust=self.gamma_adjust,
            max_patches_per_scan=self.max_patches_per_scan,
            ignore_outside=self.ignore_outside,
        )
        print(f"Number of samples in train_dataset: {len(self.train_dataset)}")

        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files, _ = self._split(files)

            rng = np.random.default_rng(self.seed)
            val_transforms = T.Compose(
                T.RelativePatchCoordinates(),
                T.IntensityAsFeatures(),
                T.PositiveNegativePatches(
                    max_patches=self.max_patches_per_scan,
                    ignore_outside=self.ignore_outside,
                    rng=rng,
                ),
                T.ToTensor(),
            )
            train_transforms = T.Compose(
                T.RandomXAxisFlip(rng=rng),
                T.RandomPatchTranslate(max_voxels=16, rng=rng),
                val_transforms,
                T.RandomGammaAdjust(rng=rng) if self.gamma_adjust else dict,
            )

            self.train_dataset = MandibleSegDataset(
                stage='fit',
                files=train_files,
                transform=train_transforms,
                **self.dataset_cfg,
            )
            self.val_dataset = MandibleSegDataset(
                stage='fit',
                files=val_files,
                transform=val_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'test':
            files = self._files('test')
            _, test_files, _ = self._split(files)

            self.test_dataset = MandibleSegDataset(
                stage='test',
                files=test_files[::-1],
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'predict':
            all_files = self._files('predict')

            non_mandible_files = []
            for files in all_files:
                mandible_file = self.root / files[0].parent / f'{files[0].stem[:-9]}.nii.gz'
                if mandible_file.exists():
                    continue
                
                non_mandible_files.append(files)

            self.predict_dataset = MandibleSegDataset(
                stage='predict',
                files=non_mandible_files[:4],
                transform=self.default_transforms,
                **self.dataset_cfg,
            )
    '''
    def setup(self, stage: Optional[str] = None) -> None:
        print(f"Setting up MandibleSegDataModule with root: {self.root}")
        print(f"Files in root directory: {os.listdir(self.root)}")

        files = self._files('fit')
        train_files, val_files, _ = self._split(files)

        rng = np.random.default_rng(self.seed)
        val_transforms = T.Compose(
            T.RelativePatchCoordinates(),
            T.IntensityAsFeatures(),
            T.PositiveNegativePatches(
                max_patches=self.max_patches_per_scan,
                ignore_outside=self.ignore_outside,
                rng=rng,
            ),
            T.ToTensor(),
        )
        train_transforms = T.Compose(
            T.RandomXAxisFlip(rng=rng),
            T.RandomPatchTranslate(max_voxels=16, rng=rng),
            val_transforms,
            T.RandomGammaAdjust(rng=rng) if self.gamma_adjust else dict,
        )

        self.train_dataset = MandibleSegDataset(
            stage='fit',
            files=train_files,
            root=self.root,
            regex_filter=self.regex_filter,
            regular_spacing=self.regular_spacing[0],
            patch_size=self.patch_size,
            stride=self.stride[0],
            gamma_adjust=self.gamma_adjust,
            max_patches_per_scan=self.max_patches_per_scan,
            ignore_outside=self.ignore_outside,
            transform=train_transforms,
        )
        print(f"Number of samples in train_dataset: {len(self.train_dataset)}")

        if stage is None or stage == 'fit':
            self.val_dataset = MandibleSegDataset(
                stage='fit',
                files=val_files,
                root=self.root,
                regex_filter=self.regex_filter,
                regular_spacing=self.regular_spacing[0],
                patch_size=self.patch_size,
                stride=self.stride[0],
                gamma_adjust=self.gamma_adjust,
                max_patches_per_scan=self.max_patches_per_scan,
                ignore_outside=self.ignore_outside,
                transform=val_transforms,
            )

        if stage is None or stage == 'test':
            test_files = self._files('test')
            self.test_dataset = MandibleSegDataset(
                stage='test',
                files=test_files,
                root=self.root,
                regex_filter=self.regex_filter,
                regular_spacing=self.regular_spacing[0],
                patch_size=self.patch_size,
                stride=self.stride[0],
                gamma_adjust=self.gamma_adjust,
                max_patches_per_scan=self.max_patches_per_scan,
                ignore_outside=self.ignore_outside,
                transform=self.default_transforms,
            )

        if stage is None or stage == 'predict':
            predict_files = self._files('predict')
            self.predict_dataset = MandibleSegDataset(
                stage='predict',
                files=predict_files,
                root=self.root,
                regex_filter=self.regex_filter,
                regular_spacing=self.regular_spacing[0],
                patch_size=self.patch_size,
                stride=self.stride[0],
                gamma_adjust=self.gamma_adjust,
                max_patches_per_scan=self.max_patches_per_scan,
                ignore_outside=self.ignore_outside,
                transform=self.default_transforms,
            )
    @property
    def num_channels(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 1

    def fit_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        Tuple[
            TensorType['P', 3, torch.float32],
            TensorType['P', 'size', 'size', 'size', torch.float32],
        ],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        features = torch.cat(batch_dict['features'])
        coords = torch.cat(batch_dict['coords'])
        labels = torch.cat(batch_dict['masks'])

        return features, (coords, labels)

    def test_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['d', 'h', 'w', 3, 2, torch.int64],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        features = batch[0]['features']
        patch_idxs = batch[0]['patch_idxs']
        labels = batch[0]['labels']
        
        return features, patch_idxs, labels

    def predict_collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['C', 'D', 'H', 'W', torch.float32],
        TensorType['d', 'h', 'w', 3, 2, torch.int64],
        TensorType[4, 4, torch.float32],
        TensorType[3, torch.int64],
    ]:
        features = batch[0]['features']
        patch_idxs = batch[0]['patch_idxs']
        affine = batch[0]['affine']
        shape = batch[0]['shape']

        return features, patch_idxs, affine, shape
