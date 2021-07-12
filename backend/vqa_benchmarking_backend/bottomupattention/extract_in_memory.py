# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
# Modified by Dirk Vaeth
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
from bottomupattention.models.bua.config import add_bottom_up_attention_config
import sys
import torch
import cv2
import os
import numpy as np
sys.path.append('detectron2')

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup

from bottomupattention.utils.extract_utils import get_image_blob
from bottomupattention.models.bua.layers.nms import nms

import ray


# class _Args:
#     config_file = "bottomupattention/configs/bua-caffe/extract-bua-caffe-r101.yaml"
#     num_cpus = 12
#     gpu_id = "2"
#     output_dir = 'bottomupattention/output'
#     bbox_dir = 'bottomupattention/output'
class _Args:
    pass


def setup(config_file: str, min_boxes: int, max_boxes: int, gpu_id: int, num_cpus: int = 0):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_bottom_up_attention_config(cfg, True)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.BUA.EXTRACTOR.MODE', 1])
    cfg.merge_from_list(['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
                         'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes])
    cfg.freeze()
    args = _Args()
    default_setup(cfg, args)

    if gpu_id > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    print("SET GPU IDs", gpu_id)
    model = _load_checkpoint(cfg)

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    # if num_cpus != 0:
    #     ray.init(num_cpus=num_cpus)
    # else:
    #     ray.init(num_gpus=1)

    return model, cfg


def _load_checkpoint(cfg):
    print('loading checkpoint', cfg.MODEL.WEIGHTS)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    model.eval()
    return model

def _parse_roi_features(cfg, im_file, im, dataset_dict, boxes, scores, features_pooled, attr_scores=None):
    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH
  
    dets = boxes[0] / dataset_dict['im_scale']
    scores = scores[0]
    feats = features_pooled[0]

    max_conf = torch.zeros((scores.shape[0])).to(scores.device)
    for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                             cls_scores[keep],
                                             max_conf[keep])
            
    keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
    image_feat = feats[keep_boxes]
    image_bboxes = dets[keep_boxes]
    image_objects_conf = np.max(scores[keep_boxes].numpy()[:,1:], axis=1)
    image_objects = np.argmax(scores[keep_boxes].numpy()[:,1:], axis=1)
    if not attr_scores is None:
        attr_scores = attr_scores[0]
        image_attrs_conf = np.max(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
        image_attrs = np.argmax(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
        info = {
            'image_id': im_file.split('.')[0],
            'image_h': np.size(im, 0),
            'image_w': np.size(im, 1),
            'num_boxes': len(keep_boxes),
            'objects_id': image_objects,
            'objects_conf': image_objects_conf,
            'attrs_id': image_attrs,
            'attrs_conf': image_attrs_conf,
            }
    else:
        info = {
            'image_id': im_file.split('.')[0],
            'image_h': np.size(im, 0),
            'image_w': np.size(im, 1),
            'num_boxes': len(keep_boxes),
            'objects_id': image_objects,
            'objects_conf': image_objects_conf
            }
    return {
        "x": image_feat,
        "bbox": image_bboxes,
        "num_bbox": len(keep_boxes),
        "image_h": np.size(im, 0),
        "image_w": np.size(im, 1),
        "info": info
    }

# @ray.remote(num_gpus=0.25, num_cpus=3)
def extract_feat_in_memory(model, im_file, cfg):
    model.eval()
    im = cv2.imread(im_file)
    if im is None:
        raise Exception(f'{im_file} is illegal!')
    dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
    # extract roi features
    attr_scores = None
    with torch.set_grad_enabled(False):
        if cfg.MODEL.BUA.ATTRIBUTE_ON:
            boxes, scores, features_pooled, attr_scores = model([dataset_dict])
        else:
            boxes, scores, features_pooled = model([dataset_dict])
    boxes = [box.tensor.cpu() for box in boxes]
    scores = [score.cpu() for score in scores]
    features_pooled = [feat.cpu() for feat in features_pooled]
    if not attr_scores is None:
        attr_scores = [attr_score.cpu() for attr_score in attr_scores]
    return _parse_roi_features(cfg, im_file, im, dataset_dict, 
        boxes, scores, features_pooled, attr_scores)
    

