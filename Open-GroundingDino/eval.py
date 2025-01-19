import os
import cv2
import argparse
import tqdm
import ujson
import torch
import numpy as np
import datasets.transforms as T

from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from groundingdino.util.inference import load_model, load_dipex, load_image, predict, annotate
from groundingdino.util.utils import get_phrases_from_posmap
from coco import CocoDetection
from typing import Tuple, List

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run inference on images")
    parser.add_argument('--caption', type=str, default="part .", help='Text query')
    parser.add_argument('--bbox-thresh', type=float, default=0.1, help='Box threshold')
    parser.add_argument('--text-thresh', type=float, default=0.1, help='Text threshold')
    parser.add_argument('--cfg-path', type=str, default="config/cfg_eval.py", help='Path to the configuration file')
    parser.add_argument('--ckpt-path', default="weights/groundingdino_swint_ogc.pth", type=str, help='Path to a pre-trained checkpoint')
    parser.add_argument('--output-path', type=str, default="output.json", help='Path to save the output JSON')
    parser.add_argument('--dataset-root', nargs='+', type=str, help='Root directory of the dataset')
    parser.add_argument('--gt-anns-path', type=str, help='Path to ground truth annotations')
    args = parser.parse_args()
    
    return args

def load_dataset(args):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = CocoDetection(roots=args.dataset_root,
                            annFile=args.gt_anns_path, 
                            transforms=transform, 
                            cache_mode=True)
    return DataLoader(dataset=dataset, 
                      batch_size=1,
                      shuffle=False,)

def untransform(image):
    """for annotating purposes only"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    
    # unnormalize the image
    image_np = image.permute(1, 2, 0).cpu().detach().numpy()  # convert to HxWxC
    image_np = (image_np * std.numpy() + mean.numpy())  # unnormalize

    # clamp values within [0, 1] before scaling
    image_np = np.clip(image_np, 0, 1)

    # scale to [0, 255] and convert to uint8
    image_np = (image_np * 255).astype(np.uint8)

    return image_np

def get_grounding_output(args, model, image, orig_size, img_id, save=True):
    boxes, score, phrases = predict(    
        model=model,
        image=image,
        caption=args.caption,
        box_threshold=args.bbox_thresh,
        text_threshold=args.text_thresh
    )

    # re-scale boxes
    scaled_boxes = boxes * torch.Tensor([*orig_size, *orig_size]).to(image.device)
    boxes_xywh = box_convert(scaled_boxes, in_fmt="cxcywh", out_fmt="xywh").detach().numpy()
    
    if save:
        image_np = untransform(image)
        annotated_frame = annotate(image_np, boxes, score, phrases)
        cv2.imwrite(f'results/annotated_{img_id.item()}.png', annotated_frame)

    # to filter-out duplicates
    results = []
    for idx, (x, y, w, h) in enumerate(boxes_xywh):
        bbox_tuple = (x, y, w, h)
        box_info = {
            "image_id": int(img_id.item()),
            "category_id": 1, # class-agnostic
            "bbox": [float(coord) for coord in bbox_tuple],
            "score": float(score[idx]),
            "category_name": phrases[idx]
        }
        results.append(box_info)
    return results

def load_json(path=None):
    assert path != None

    with open(path, 'r') as file:
        json_data = ujson.load(file)

    return json_data

def run_eval(coco_gt, res=None, cat_ids=[]):
    assert res is not None
    
    pred = load_json(path=res)
    coco_pred = coco_gt.loadRes(pred)

    image_ids = [ann['image_id'] for ann in coco_pred.anns.values()]
    unique_ids, _ = np.unique(image_ids, return_counts=True)
    
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    # coco_eval.params.maxDets = [1,10,100]
    # coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(f"total: {len(unique_ids)} images")
    print(f"dets/img: {len(image_ids)/len(unique_ids)}")

def run_inference(args, coco_gt, dataloader, viz=False):
    model = load_dipex(args.cfg_path, args.ckpt_path)
    with open(args.output_path, 'w') as file:
        file.write('[')
        first_entry = True
        for index, item in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            if item is None:  # skip if none
                continue
            img_tensors, targets, orig_size, img_id = item
            batch_results = get_grounding_output(args, model, img_tensors[0], orig_size, img_id, save=False)
            for item in batch_results:
                if not first_entry:
                    file.write(',')
                else:
                    first_entry = False

                ujson.dump(item, file)

        file.write(']')

def main():
    args = parse_args()
    dataloader = load_dataset(args)
    coco_gt = COCO(args.gt_anns_path)
    run_inference(args, coco_gt, dataloader)
    run_eval(coco_gt, "output.json")

if __name__ == '__main__':
    main()