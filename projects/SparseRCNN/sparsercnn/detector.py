#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.matcher import Matcher
from detectron2.utils.visualizer import ColorMode, Visualizer, VisImage

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.logger import log_first_n
from detectron2.utils.events import get_event_storage
from fvcore.nn import giou_loss, smooth_l1_loss

import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure

from .loss import SetCriterion, HungarianMatcher, WSDDNCriterion, OICRCriterion
from .head import DynamicHead, DynamicHeadWSL, BoxHead, WSDDNOutputLayers, OICROutputLayers
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import os
import pdb

__all__ = ["SparseRCNN", "SparseRCNNWSL"]


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return (
        [x[0] for x in result_per_image],
        [x[1] for x in result_per_image],
        [x[2] for x in result_per_image],
        [x[3] for x in result_per_image],
    )


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """

    all_scores = scores.clone()
    all_scores = torch.unsqueeze(all_scores, 0)
    all_boxes = boxes.clone()
    all_boxes = torch.unsqueeze(all_boxes, 0)

    pred_inds = torch.unsqueeze(
        torch.arange(scores.size(0), device=scores.device, dtype=torch.long), dim=1
    ).repeat(1, scores.size(1))

    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        pred_inds = pred_inds[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    pred_inds = pred_inds[:, :-1]

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    pred_inds = pred_inds[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    pred_inds = pred_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.pred_inds = pred_inds
    return result, filter_inds[:, 0], all_scores, all_boxes

@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh


@META_ARCH_REGISTRY.register()
class SparseRCNNWSDDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.box_head_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # Build Weakly Supervised learning head.
        self.box_head = BoxHead(cfg)
        self.box_predictor = WSDDNOutputLayers(cfg)
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT

        # Build Proposals.  (Uniform Initial)
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        
        # 1. 完全随机初始化
        # cx, cy = torch.rand(self.num_proposals, 1), torch.rand(self.num_proposals, 1)
        # w, h = torch.rand(self.num_proposals, 1), torch.rand(self.num_proposals, 1)
        # w = torch.clamp_min(w, 1e-4)
        # h = torch.clamp_min(h, 1e-4)
        # max_w = torch.where(cx < 0.5, 2*cx, 2-2*cx)
        # max_h = torch.where(cy < 0.5, 2*cy, 2-2*cy)
        # w = torch.where(w < max_w, w, max_w)
        # h = torch.where(h < max_h, h, max_h)
        # init_proposal_boxes = torch.cat([cx, cy, w, h], dim=1)
        # self.init_proposal_boxes.weight = nn.Parameter(init_proposal_boxes)

        # 2. 均匀分布初始化
        # proposal_num_root = int(self.num_proposals ** 0.5)
        # xx = [1 / proposal_num_root * i + 0.5 / proposal_num_root for i in range(proposal_num_root)]
        # yy = [1 / proposal_num_root * i + 0.5 / proposal_num_root for i in range(proposal_num_root)]
        # xy = []
        # for x in xx:
        #     for y in yy:
        #         xy.append([x,y])
        # wh = [[1/proposal_num_root, 1/proposal_num_root] for _ in range(proposal_num_root ** 2)]
        # init_proposal_boxes = torch.cat([torch.tensor(xy), torch.tensor(wh)], 1)
        # self.init_proposal_boxes.weight = nn.Parameter(init_proposal_boxes)

        # 3. 密集均匀分布初始化
        proposal_num_root = int((self.num_proposals / 9) ** 0.5)
        xx = [1 / proposal_num_root * i + 0.5 / proposal_num_root for i in range(proposal_num_root)]
        yy = [1 / proposal_num_root * i + 0.5 / proposal_num_root for i in range(proposal_num_root)]
        xy = []
        for x in xx:
            for y in yy:
                xy.append([x,y])
        sizes = [1/4, 2/4, 3/4]
        aspect_ratios = [0.5, 1, 2]
        init_proposal_boxes = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area /aspect_ratio)
                h = aspect_ratio * w
                for _xy in xy:
                    init_proposal_boxes.append([_xy[0], _xy[1], w, h])
        init_proposal_boxes = torch.tensor(init_proposal_boxes)
        # self.init_proposal_boxes.weight = nn.Parameter(init_proposal_boxes)
        # !!!! 固定框位置
        self.init_proposal_boxes.weight = nn.Parameter(init_proposal_boxes, False)


        # 4. 参照sparsercnn初始化
        # nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        # nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.dynamic_head = DynamicHeadWSL(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION



        # Build Criterion
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        self.criterion = WSDDNCriterion(
            cfg=cfg,
            matcher=matcher
        )

        # test
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        # Others
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.save_dir = cfg.OUTPUT_DIR
        self.step = 0
        self.box_pooler = DynamicHeadWSL(cfg=cfg, roi_input_shape=self.backbone.output_shape())._init_box_pooler(cfg, self.backbone.output_shape())


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
        
        # 忽略！
        # 方案1 直接用类似rpn生成的proposal来做roi pooling
        METHOD_1 = True
        if METHOD_1:
            roi_features = self.box_pooler(features, [Boxes(bbox) for bbox in proposal_boxes]).unsqueeze(0)
            box_features = self.box_head(roi_features)                  # [num_head, bn*num_box, box_head_dim]
            box_features = box_features.view(self.num_heads, -1, self.num_proposals, self.box_head_dim) # [num_head, bn, num_box, box_head_dim]
            wsddn_pred_logits = self.box_predictor(box_features)        # [num_head, bn, num_box, num_cls]
            output = {'pred_logits': wsddn_pred_logits[0]}
            if self.training:    
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances)
                loss_dict = self.criterion(output, targets, None, None, 0)
                return loss_dict
            else:
                box_cls = output["pred_logits"]
                box_pred = proposal_boxes
                results = self.inference(box_cls, box_pred, images.image_sizes)
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})

                return processed_results

        # Prediction.
        outputs_class, outputs_coord, roi_features, proposals, pred_proposal_deltas = self.dynamic_head(features, proposal_boxes, self.init_proposal_features.weight)

        # WSOD
        if self.deep_supervision:
            box_features = self.box_head(roi_features)                  # [num_head, bn*num_box, box_head_dim]
            box_features = box_features.view(self.num_heads, -1, self.num_proposals, self.box_head_dim) # [num_head, bn, num_box, box_head_dim]
            wsddn_pred_logits = self.box_predictor(box_features)        # [num_head, bn, num_box, num_cls]

        else:
            box_features = self.box_head(roi_features[-1][None])  # [1, bn * num_box, box_head_dim]
            box_features = box_features.view(1, -1, self.num_proposals, self.box_head_dim) # [1, bn, num_box, box_head_dim]
            wsddn_pred_logits = self.box_predictor(box_features) # [1, bn, num_box, num_cls]
            # TODOO
        
        multihead_outputs = [{"pred_logits": a, "pred_boxes": b, "proposal_boxes": c, "pred_proposal_deltas": d} \
            for a, b, c, d in zip(wsddn_pred_logits, outputs_coord, proposals, pred_proposal_deltas)]

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = {}
            if self.deep_supervision:
                # 遍历每个dynamic head
                for i, output in enumerate(multihead_outputs):
                    # 查看最大值
                    self.step += 1
                    # print('===step:{}'.format(self.step))
                    # print('max:{:.3f} min:{:.3f}'.format(output['pred_proposal_deltas'].max().item(), output['pred_proposal_deltas'].min().item()))

                    proposals = []
                    for proposals_per_image, gt_instance in zip(output["proposal_boxes"], gt_instances):
                        proposal = Instances(gt_instance.image_size)
                        proposal.set("proposal_boxes", Boxes(proposals_per_image))
                        proposal_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
                        proposal_logits = proposal_logit_value * torch.ones(len(proposal.proposal_boxes), device=gt_instance.gt_boxes.device)
                        proposal.set("objectness_logits", proposal_logits)
                        proposals.append(proposal)
                    
                    # pseudo_targets = self.get_pgt_top_k(output["pred_boxes"].detach(), output["pred_logits"].detach(), targets, gt_instances)
                    pseudo_targets, pgt_inds = self.get_pgt_top_k(output["pred_boxes"].detach(), output["pred_logits"].detach(), targets, gt_instances, return_ind=True)

                    proposals = self.label_and_sample_proposals(proposals, pseudo_targets)
                    loss_dict.update(self.criterion(output, targets, proposals, pseudo_targets, i))
                    
                    SAVE_IMG = False
                    if SAVE_IMG:
                    # if not self.step % 50:
                        save_dir = os.path.join(self.save_dir, 'visual')
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        # 遍历batch中每个图片
                        for img_input, pseudo_target, anchors, proposal, pgt_ind in zip(batched_inputs, pseudo_targets, output['proposal_boxes'], output["pred_boxes"], pgt_inds):
                            img_ind = img_input['file_name'].split('/')[-1].split('.')[0]

                            OLD_DRAW = False
                            if OLD_DRAW:
                                vis_gt = Instances(img_input['instances'].image_size)
                                vis_gt.set('pred_boxes', img_input['instances'].gt_boxes)
                                visualizer = Visualizer(img_input['image'].permute(1, 2, 0).numpy())
                                vis_out = visualizer.draw_instance_predictions(vis_gt)
                                vis_out.save(os.path.join(save_dir, '{}_gt_{}'.format(img_ind, i)))

                                vis_pgt = Instances(img_input['instances'].image_size)
                                vis_pgt.set('pred_boxes', Boxes(pseudo_target.gt_boxes.tensor.cpu()))
                                visualizer = Visualizer(img_input['image'].permute(1, 2, 0).numpy())
                                vis_out = visualizer.draw_instance_predictions(vis_pgt)
                                # vis_out.save(os.path.join(save_dir, '{}_pgt_{}'.format(img_ind, i)))
                                vis_out.save(os.path.join(save_dir, 'step_{:04d}_{}_pgt_{}'.format(self.step, img_ind, i)))

                            if True:
                                # 画label对应的三个框
                                visImage = VisImage(img_input['image'].permute(1, 2, 0).cpu().numpy())
                                gt_boxes = img_input['instances'].gt_boxes.tensor.numpy()
                                pgt_boxes = pseudo_target.gt_boxes.tensor.cpu().numpy()
                                pgt_ind = pgt_ind.reshape(-1, 4)
                                anchors = anchors.detach().cpu().numpy()
                                proposal = proposal.detach().cpu().numpy()
                                num_instance = len(gt_boxes)
                                for j in range(num_instance):
                                    # 画红色的gt
                                    self._draw_box(visImage, gt_boxes[j], edge_color='r')
                                    # 画蓝色的伪gt
                                    self._draw_box(visImage, pgt_boxes[j], edge_color='b')
                                    # 画绿色的anchor
                                    self._draw_box(visImage, anchors[pgt_ind[j][0]], edge_color='g')
                                # visImage.save(os.path.join(save_dir, '{}_gt_{}'.format(img_ind, i)))
                                visImage.save(os.path.join(save_dir, 'step_{:04d}_{}_gt_{}'.format(self.step, img_ind, i)))

                                # 画采样的anchor和对应的proposal
                                visImage = VisImage(img_input['image'].permute(1, 2, 0).cpu().numpy())
                                SAMPLE_NUM = 8
                                COLOR_SET = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
                                sample_ind = np.random.randint(0, len(proposal), (SAMPLE_NUM))

                                # # 取400个框看看是否足够dense
                                # sample_ind_all = np.random.randint(0, len(proposal), (400))
                                # for j, sample_proposal in enumerate(proposal[sample_ind_all]):
                                #     self._draw_box(visImage, sample_proposal, edge_color=COLOR_SET[j%len(COLOR_SET)])
                                # visImage.save(os.path.join(save_dir, 'step_{:04d}_bbox_all_{}'.format(self.step, img_ind, i)))

                                for j, (sample_proposal, sample_anchor) in enumerate(zip(proposal[sample_ind], anchors[sample_ind])):
                                    self._draw_box(visImage, sample_proposal, edge_color=COLOR_SET[j%len(COLOR_SET)])
                                    self._draw_box(visImage, sample_anchor, edge_color=COLOR_SET[j%len(COLOR_SET)])
                                # visImage.save(os.path.join(save_dir, '{}_bbox_match_{}'.format(img_ind, i)))
                                visImage.save(os.path.join(save_dir, 'step_{:04d}_bbox_match_{}'.format(self.step, img_ind, i)))
                    
            return loss_dict

        else:
            box_cls = multihead_outputs[-1]["pred_logits"]
            box_pred = multihead_outputs[-1]["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

    def _draw_box(self, output, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        # linewidth = max(self._default_font_size / 4, 1)
        linewidth = 2

        output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=line_style,
            )
        )

    def get_pgt_top_k(self, prev_pred_boxes, prev_pred_scores, targets, gt_instances, top_k=1, return_ind=False):
        # 选择标签类别对应的所有box和score
        batch_size = len(targets)
        prev_pred_img_scores = prev_pred_scores.sum(dim=1)   # [bs, num_cls]
        prev_pred_img_scores = torch.clamp(prev_pred_img_scores, min=1e-6, max=1.0 - 1e-6)
        prev_pred_scores = [      # prev_pred_score: bs * [num_proposal, gt_num_cls]
            torch.index_select(prev_pred_score, 1, target['labels']) for (prev_pred_score, target) in zip(prev_pred_scores, targets)
        ]
        prev_pred_boxes = prev_pred_boxes.unsqueeze(2).expand(batch_size, self.num_proposals, self.num_classes, 4)
        prev_pred_boxes = [       # prev_pred_boxes: bs * [num_proposal, gt_num_cls, 4]
            torch.index_select(prev_pred_box, 1, target['labels']) for (prev_pred_box, target) in zip(prev_pred_boxes, targets)
        ]

        # 每个类别选出topk个box作pseudo label, 目前为topk = 1
        num_preds = [self.num_proposals] * batch_size
        if top_k >= 1:
            top_ks = [min(num_pred, int(top_k)) for num_pred in num_preds]
        elif top_k < 1 and top_k > 0:
            top_ks = [max(int(num_pred * top_k), 1) for num_pred in num_preds]
        else:
            top_ks = [min(num_pred, 1) for num_pred in num_preds]
        
        # 每个标签类别计算topk个box的index
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]

        pgt_scores = [item[0] for item in pgt_scores_idxs]   # bs * [gt_num_cls]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]     # bs * [gt_num_cls]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, target['labels'].numel(), 4) 
            for pgt_idx, top_k, target in zip(pgt_idxs, top_ks, targets)
        ]                                                    # bs * [topk(1), gt_num_cls, 4]

        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]                                                    # bs * [topk(1), gt_num_cls, 4]
        pgt_classes = [
            torch.unsqueeze(target['labels'], 0).expand(top_k, target['labels'].numel())
            for target, top_k in zip(targets, top_ks)
        ]                                                    # bs * [topk(1), gt_num_cls]
        pgt_weights = [
            torch.index_select(pred_logits, 1, target['labels']).expand(top_k, target['labels'].numel())
            for pred_logits, target, top_k in zip(
                prev_pred_img_scores.split(1, dim=0), targets, top_ks
            )
        ]                                                    # bs * [topk(1), gt_num_cls]


        # reshape
        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]
        targets = [
            Instances(
                gt_instances[i].image_size,
                gt_boxes=pgt_box,
                gt_classes=pgt_class,
                gt_scores=pgt_score,
                gt_weights=pgt_weight,
            )
            for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights)
            )
        ]
        if return_ind:
             return targets, pgt_idxs
        return targets

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], suffix=""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]
                # alpha = 1 - 1.0 * self.iter / self.max_iter
                # proposals_per_image.gt_weights = (1 - alpha) * proposals_per_image.gt_weights + alpha * proposals_per_image.objectness_logits
                # proposals_per_image.gt_weights = torch.clamp(proposals_per_image.gt_weights, min=1e-6, max=1.0 - 1e-6)
            if has_gt and targets_per_image.has("gt_masks"):
                proposals_per_image.gt_masks = targets_per_image.gt_masks[
                    matched_idxs[sampled_idxs]
                ]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes[sampled_idxs]

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # assert len(box_cls) == len(image_sizes)
        # results = []

        # # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        # for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
        #     scores, labels, box_pred, image_sizes
        # )):
        #     result = Instances(image_size)
        #     result.pred_boxes = Boxes(box_pred_per_image)   # [800, 4]
        #     result.scores = scores_per_image                # [800]
        #     result.pred_classes = labels_per_image          # [800]
        #     results.append(result)

        # return results

        
        # exchange h and w
        # image_sizes = [(size[1], size[0]) for size in image_sizes] # MARK!!!!

        boxes = [torch.cat([box]*20, 1) for box in box_pred]
        scores = []
        for score in box_cls:
            score_bg = torch.zeros(score.shape[0], 1, dtype=score.dtype, device=score.device, requires_grad=False)
            score = torch.cat((score, score_bg), 1)
            scores.append(score)
        results, _, all_scores, all_boxes = fast_rcnn_inference(boxes, scores, image_sizes, self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        # images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        # images = ImageList.from_tensors(images, self.size_divisibility)

        # 修改
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh


@META_ARCH_REGISTRY.register()
class SparseRCNNOICR(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # Build Proposals.  (Uniform Initial)
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        cx, cy = torch.rand(self.num_proposals, 1), torch.rand(self.num_proposals, 1)
        w, h = torch.rand(self.num_proposals, 1), torch.rand(self.num_proposals, 1)
        max_w = torch.where(cx < 0.5, 2*cx, 2-2*cx)
        max_h = torch.where(cy < 0.5, 2*cy, 2-2*cy)
        w = torch.where(w < max_w, w, max_w)
        h = torch.where(h < max_h, h, max_h)
        init_proposal_boxes = torch.cat([cx, cy, w, h], dim=1)
        self.init_proposal_boxes.weight = nn.Parameter(init_proposal_boxes)
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS
        )

        # Build Dynamic Head.
        self.dynamic_head = DynamicHeadWSL(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        
        # Build Weakly Supervised Learning Head.
        
        self.box_head = BoxHead(cfg)
        self.box_predictor = WSDDNOutputLayers(cfg)
        self.box_refinery = []
        self.K = 3
        for k in range(self.K):
            box_refinery = OICROutputLayers(cfg)
            self.add_module('box_refinery_{}'.format(k), box_refinery)
            self.box_refinery.append(box_refinery)

        # Build Criterion.
        self.wsddn_criterion = WSDDNCriterion(cfg)
        self.oicr_criterion = OICRCriterion(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord, proposal_features = self.dynamic_head(features, proposal_boxes, self.init_proposal_features.weight)
        proposal_features = proposal_features.view(len(batched_inputs), self.num_proposals, -1)

        # WSOD
        box_features = self.box_head(proposal_features)      # mark output size
        pred_logits = self.box_predictor(box_features)
        
        self.pred_class_img_logits = pred_logits.sum(dim=1)
        self.pred_class_img_logits = torch.clamp(self.pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.wsddn_criterion(pred_logits, targets)

            prev_pred_scores = pred_logits.detach()    # [bn, num_proposal, num_cls]
            prev_pred_boxes = proposal_boxes.detach()  # [bn, num_proposal, 4]
            proposals = [
                Instances(gt_instance.image_size, proposal_boxes=Boxes(proposal_box)) for proposal_box, gt_instance in zip(proposal_boxes, gt_instances)
            ]

            for k in range(self.K):
                targets = self.get_pgt_top_k(prev_pred_boxes, prev_pred_scores, targets, gt_instances)
                proposals_k = self.label_and_sample_proposals(proposals, targets)
                predictions_k = self.box_refinery[k](box_features)
                losses_k = self.OICRCriterion(predictions_k, proposals_k)
                loss_dict.update(losses_k)

                prev_pred_scores = F.softmax(prev_pred_scores, dim=-1).detach()
                prev_pred_boxes = [proposal_k.proposal_boxes.tensor.detach() for proposal_k in proposals_k]
                prev_pred_boxes = torch.cat(prev_pred_boxes).reshape(-1, self.num_proposals, 4)    

            return loss_dict

        else:
            box_cls = pred_logits
            box_pred = proposal_boxes
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]):
        gt_boxes = [x.gt_boxes for x in targets]
        self.proposal_append_gt = False
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes[sampled_idxs]

    def get_pgt_top_k(self, prev_pred_boxes, prev_pred_scores, targets, gt_instances, top_k=1):
        batch_size = len(prev_pred_boxes)
        prev_pred_boxes = prev_pred_boxes.unsqueeze(2).expand(batch_size, self.num_proposals, self.num_classes, 4)
        prev_pred_scores = [      # prev_pred_score: bn * [num_proposal, gt_num_cls]
            torch.index_select(prev_pred_score, 1, target['labels']) for (prev_pred_score, target) in zip(prev_pred_scores, targets)
        ]
        prev_pred_boxes = [       # prev_pred_boxes: bn * [num_proposal, gt_num_cls, 4]
            torch.index_select(prev_pred_box, 1, target['labels']) for (prev_pred_box, target) in zip(prev_pred_boxes, targets)
        ]
        num_preds = [self.num_proposals] * batch_size

        # select top k proposals as pseudo labels
        if top_k >= 1:
            top_ks = [min(num_pred, int(top_k)) for num_pred in num_preds]
        elif top_k < 1 and top_k > 0:
            top_ks = [max(int(num_pred * top_k), 1) for num_pred in num_preds]
        else:
            top_ks = [min(num_pred, 1) for num_pred in num_preds]
        
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, target['labels'].numel(), 4)
            for pgt_idx, top_k, target in zip(pgt_idxs, top_ks, targets)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(target['labels'], 0).expand(top_k, target['labels'].numel())
            for target, top_k in zip(targets, top_ks)
        ]
        pgt_weights = [
            torch.index_select(pred_logits, 1, target['labels']).expand(top_k, target['labels'].numel())
            for pred_logits, target, top_k in zip(
                self.pred_class_img_logits.split(1, dim=0), targets, top_ks
            )
        ]
        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(
                gt_instances[i].image_size,
                gt_boxes=pgt_box,
                gt_classes=pgt_class,
                gt_scores=pgt_score,
                gt_weights=pgt_weight,
            )
            for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights)
            )
        ]
        return targets

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
