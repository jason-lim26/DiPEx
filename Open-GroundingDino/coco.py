# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
import os
import os.path as osp
from io import BytesIO
from typing import Any, Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import tqdm
from pycocotools.coco import COCO
from functools import lru_cache


class CocoDetection(VisionDataset):
    """
    MS Coco Detection Dataset with Caching Feature:
    - Image caching to memory for faster access
    - Support for multiple root directories
    - Support for distributed processing

    Args:
        roots (str or List[str]): Root directory or list of directories where images are stored.
        annFile (str): Path to the JSON annotation file.
        transform (Callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (Callable, optional): A function/transform that takes in the target and transforms it.
        transforms (Callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
        cache_mode (bool, optional): If True, images are cached in memory.
        local_rank (int, optional): The rank of the current process in distributed training.
        local_size (int, optional): The total number of processes in distributed training.
    """

    def __init__(
        self,
        roots: Any,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        cache_mode: bool = False,
        local_rank: int = 0,
        local_size: int = 1,
    ):
        super(CocoDetection, self).__init__(roots, transforms, transform, target_transform)
        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys())
        self.roots = roots if isinstance(roots, list) else [roots]
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size

        # initialize cache if caching is enabled
        self.cache = {} if cache_mode else None
        if self.cache_mode:
            self._cache_images()
    
    def determine_root(self, file_name: str) -> str:
        """
        Determine the root directory that contains the given file.

        Args:
            file_name (str): The name of the file to locate.

        Returns:
            str: The root directory containing the file.

        Raises:
            FileNotFoundError: If the file is not found in any of the root directories.
        """
        for root in self.roots:
            if osp.exists(osp.join(root, file_name)):
                return root
        raise FileNotFoundError(f"{file_name} not found in provided root directories.")

    def _cache_images(self):
        """
        Cache images in memory. This method is called during initialization if caching is enabled.
        """
        print("Caching images in memory...")
        for index in tqdm.tqdm(range(len(self.ids)), desc="Caching images"):
            if index % self.local_size != self.local_rank:
                continue
            img_id = self.ids[index]
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            try:
                root = self.determine_root(file_name)
                with open(osp.join(root, file_name), 'rb') as f:
                    self.cache[file_name] = f.read()
            except FileNotFoundError:
                print(f"Warning: File {osp.join(root, file_name)} not found and will be skipped.")
                continue

    def _load_image_from_cache(self, file_name: str) -> Optional[Image.Image]:
        """
        Load an image from the cache.

        Args:
            file_name (str): The name of the image file.

        Returns:
            Optional[Image.Image]: The loaded PIL Image or None if not found.
        """
        if not self.cache_mode:
            return self._load_image_from_disk(file_name)

        if file_name not in self.cache:
            try:
                root = self.determine_root(file_name)
                with open(osp.join(root, file_name), 'rb') as f:
                    self.cache[file_name] = f.read()
            except FileNotFoundError:
                print(f"Warning: File {osp.join(root, file_name)} not found and will be skipped.")
                return None

        try:
            return Image.open(BytesIO(self.cache[file_name])).convert('RGB')
        except Exception as e:
            print(f"Error loading image {file_name}: {e}")
            return None
    
    def _load_image_from_disk(self, file_name: str) -> Optional[Image.Image]:
        """
        Load an image from disk.

        Args:
            file_name (str): The name of the image file.

        Returns:
            Optional[Image.Image]: The loaded PIL Image or None if not found.
        """
        try:
            root = self.determine_root(file_name)
            return Image.open(osp.join(root, file_name)).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File {osp.join(root, file_name)} not found and will be skipped.")
            return None
        except Exception as e:
            print(f"Error loading image {file_name}: {e}")
            return None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, Tuple[int, int], int]:
        """
        Retrieve an image and its annotations.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Tuple containing:
                - Transformed image tensor.
                - Target dictionary with annotations.
                - Original image size as (height, width).
                - Image ID.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # extract bounding boxes
        boxes = [obj["bbox"] for obj in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # extract labels
        labels = [obj["category_id"] for obj in annotations]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # extract additional annotation data
        areas = [obj.get("area", 0) for obj in annotations]
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = [obj.get("iscrowd", 0) for obj in annotations]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # prepare target dictionary
        target = {
            "image_id": img_id,
            "boxes": boxes,
            "labels": labels,
            "areas": areas,
            "iscrowd": iscrowd,
        }

        # get image file name using
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']

        # load image
        img = self._load_image_from_cache(file_name)
        if img is None:
            # handle missing image by raising an exception
            raise FileNotFoundError(f"Image {file_name} not found.")

        # get original image size (height, width)
        orig_size = img.size  # (width, height)
        target["orig_size"] = torch.as_tensor([orig_size[1], orig_size[0]])

        # apply transforms if any
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, orig_size, img_id

    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.ids)

    def image_exists(self, img_id: int) -> bool:
        """
        Check if an image exists in the dataset.

        Args:
            img_id (int): Image ID.

        Returns:
            bool: True if the image exists, False otherwise.
        """
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        try:
            self.determine_root(file_name)
            return True
        except FileNotFoundError:
            return False

    def filter_missing_images(self):
        """
        Filter out images that are missing from the filesystem.
        """
        print("Filtering out missing images...")
        original_len = len(self.ids)
        self.ids = [img_id for img_id in self.ids if self.image_exists(img_id)]
        filtered_len = len(self.ids)
        print(f"Filtered {original_len - filtered_len} missing images.")
