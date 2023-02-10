import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle as pkl
from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from ...ops.iou3d_nms import iou3d_nms_utils


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)
        if nms_config.BOX_FINE_TUNE.ENABLED:
            rois_f = batch_box_preds.new_zeros((batch_size, 2 * nms_config.BOX_FINE_TUNE.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
            roi_scores_f = batch_box_preds.new_zeros((batch_size, 2 * nms_config.BOX_FINE_TUNE.NMS_POST_MAXSIZE))
            roi_labels_f = batch_box_preds.new_zeros((batch_size, 2 * nms_config.BOX_FINE_TUNE.NMS_POST_MAXSIZE), dtype=torch.long)
            len_list = []
        else:
            len_list = [0] * batch_size
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)
            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

            len_0 = 0
            len_1 = 0
            if nms_config.BOX_FINE_TUNE.ENABLED:
                selected_f, selected_scores_f = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config.BOX_FINE_TUNE
                )
                # s: original, f: fine tune
                s_0, s_1 = box_preds[selected][cur_roi_labels[selected] == 0], box_preds[selected][cur_roi_labels[selected] == 1]
                f_0, f_1 = box_preds[selected_f][cur_roi_labels[selected_f] == 0], box_preds[selected_f][cur_roi_labels[selected_f] == 1]
                labels_0, labels_1 = cur_roi_labels[selected_f][cur_roi_labels[selected_f] == 0], cur_roi_labels[selected_f][cur_roi_labels[selected_f] == 1]
                scores_0, scores_1 = cur_roi_scores[selected_f][cur_roi_labels[selected_f] == 0], cur_roi_scores[selected_f][cur_roi_labels[selected_f] == 1]

                if len(s_0) > 1 and len(f_0) > 0:
                    ious_0 = iou3d_nms_utils.boxes_iou3d_gpu(f_0, s_0)
                    s_f_0 = s_0[torch.topk(ious_0, k=2).indices[:, 1]]
                    mask_0 = torch.topk(ious_0, k=2).values[:, 1] >= nms_config.BOX_FINE_TUNE.MASK_THRESH
                    len_0 = mask_0.sum().cpu() .numpy()
                    if len_0 > 0:
                        la = nms_config.BOX_FINE_TUNE.LAMBDA
                        rois_f[index, :len_0, :3] = s_f_0[mask_0][:, :3] * la + f_0[mask_0][:, :3] * (1-la)
                        rois_f[index, len_0:2*len_0, :3] = -s_f_0[mask_0][:, :3] * la + f_0[mask_0][:, :3] * (1 + la)
                        lwh_0 = torch.cat([s_f_0[mask_0][:, 3:6].reshape(1, -1), f_0[mask_0][:, 3:6].reshape(1, -1)]).max(dim=0).values.reshape(-1, 3)
                        rois_f[index, :2*len_0, 3:6] = torch.cat([lwh_0, lwh_0])
                        rois_f[index, :2*len_0, 6:7] = torch.cat([f_0[mask_0][:, 6:7], f_0[mask_0][:, 6:7]])
                        roi_scores_f[index, :2*len_0] = torch.cat([scores_0[mask_0], scores_0[mask_0]])
                        roi_labels_f[index, :2*len_0] = torch.cat([labels_0[mask_0], labels_0[mask_0]])

                if len(s_1) > 1 and len(f_1) > 0:
                    ious_1 = iou3d_nms_utils.boxes_iou3d_gpu(f_1, s_1)
                    s_f_1 = s_1[torch.topk(ious_1, k=2).indices[:, 1]]
                    mask_1 = torch.topk(ious_1, k=2).values[:, 1] >= nms_config.BOX_FINE_TUNE.MASK_THRESH
                    len_1 = mask_1.sum().cpu() .numpy()
                    if len_1 > 0:
                        la = nms_config.BOX_FINE_TUNE.LAMBDA
                        rois_f[index, 2*len_0:2*len_0+len_1, :3] = s_f_1[mask_1][:, :3] * la + f_1[mask_1][:, :3] * (1-la)
                        rois_f[index, 2*len_0+len_1:2*len_0+2*len_1, :3] = -s_f_1[mask_1][:, :3] * la + f_1[mask_1][:, :3] * (1 + la)
                        lwh_1 = torch.cat([s_f_1[mask_1][:, 3:6].reshape(1, -1), f_1[mask_1][:, 3:6].reshape(1, -1)]).max(dim=0).values.reshape(-1, 3)
                        rois_f[index, 2*len_0:2*len_0+2*len_1, 3:6] = torch.cat([lwh_1, lwh_1])
                        rois_f[index, 2*len_0:2*len_0+2*len_1, 6:7] = torch.cat([f_1[mask_1][:, 6:7], f_1[mask_1][:, 6:7]])
                        roi_scores_f[index, 2*len_0:2*len_0+2*len_1] = torch.cat([scores_1[mask_1], scores_1[mask_1]])
                        roi_labels_f[index, 2*len_0:2*len_0+2*len_1] = torch.cat([labels_1[mask_1], labels_1[mask_1]])
                len_list.append(2 * nms_config.BOX_FINE_TUNE.NMS_POST_MAXSIZE - 2 * len_0 - 2 * len_1)
        if nms_config.BOX_FINE_TUNE.ENABLED:
            rois = torch.cat([rois, rois_f], dim=1)
            roi_scores = torch.cat([roi_scores, roi_scores_f], dim=1)
            roi_labels = torch.cat([roi_labels, roi_labels_f], dim=1)

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['len_list'] = len_list
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_intra_triplet_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        tb_dict = {}
        tb_dict['intra_triplet_loss'] = 0
        if not self.model_cfg.TRIPLET_LOSS.INTRA_ENABLED:
            return 0, tb_dict
        rcnn_tri = forward_ret_dict['rcnn_tri']
        gt_boxes3d_class = forward_ret_dict['gt_of_rois'][..., -1].view(-1)
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        if self.model_cfg.TRIPLET_LOSS.FG_MASK:
            fg_mask = (reg_valid_mask > 0).view(-1)
            rcnn_tri = rcnn_tri[fg_mask]
            gt_boxes3d_class = gt_boxes3d_class[fg_mask]
        class_mask_1 = (gt_boxes3d_class == 1)
        class_mask_2 = (gt_boxes3d_class == 2)
        input1, input2 = rcnn_tri[class_mask_1], rcnn_tri[class_mask_2]
        if (len(input1) == 0) or (len(input2) == 0):
            return 0, tb_dict
        else:
            dist1, dist2 = torch.cdist(input1, input1, p=2),torch.cdist(input2, input2, p=2)
            dist12 = torch.cdist(input1, input2, p=2)
            p_1, n_1 = input1[dist1.max(dim=0).indices], input2[dist12.min(dim=1).indices]
            p_2, n_2 = input2[dist2.max(dim=0).indices], input1[dist12.min(dim=0).indices]
            triplet_loss_intra = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
            intra_triplet_loss = triplet_loss_intra(input1, p_1, n_1) + triplet_loss_intra(input2, p_2, n_2)
            if self.model_cfg.TRIPLET_LOSS.INTER_ENABLED:
                ROOT_DIR = self.model_cfg.TRIPLET_LOSS.EMBEDDING_DIR
                embedding = {}
                embedding['input1'], embedding['input2'] = input1, input2
                ps_path = os.path.join(ROOT_DIR, "{}_embedding_cuda{}.pkl".format(forward_ret_dict['dataset_name'], input1.device.index))
                with open(ps_path, 'wb') as f:
                    pkl.dump(embedding, f)
        intra_triplet_loss = intra_triplet_loss * loss_cfgs.LOSS_WEIGHTS['intra_triplet_loss']
        tb_dict['intra_triplet_loss'] = intra_triplet_loss.item() if intra_triplet_loss != 0 else 0
        return intra_triplet_loss, tb_dict

    def get_inter_triplet_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        tb_dict = {}
        tb_dict['inter_triplet_loss'] = 0
        if not self.model_cfg.TRIPLET_LOSS.INTER_ENABLED:
            return 0, tb_dict
        if not self.model_cfg.TRIPLET_LOSS.SRC_GRAD and forward_ret_dict['dataset_name'] == self.model_cfg.TRIPLET_LOSS.DATASETS_NAME[0]:
            return 0, tb_dict
        rcnn_tri = forward_ret_dict['rcnn_tri']
        gt_boxes3d_class = forward_ret_dict['gt_of_rois'][..., -1].view(-1)
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        if self.model_cfg.TRIPLET_LOSS.FG_MASK:
            fg_mask = (reg_valid_mask > 0).view(-1)
            rcnn_tri = rcnn_tri[fg_mask]
            gt_boxes3d_class = gt_boxes3d_class[fg_mask]
        class_mask_1 = (gt_boxes3d_class == 1)
        class_mask_2 = (gt_boxes3d_class == 2)
        input1, input2 = rcnn_tri[class_mask_1], rcnn_tri[class_mask_2]
        try:
            ROOT_DIR = self.model_cfg.TRIPLET_LOSS.EMBEDDING_DIR
            if forward_ret_dict['dataset_name'] == self.model_cfg.TRIPLET_LOSS.DATASETS_NAME[0]:
                embedding_path = os.path.join(ROOT_DIR, "{}_embedding_cuda{}.pkl".format(self.model_cfg.TRIPLET_LOSS.DATASETS_NAME[1], input1.device.index))
            else:
                embedding_path = os.path.join(ROOT_DIR, "{}_embedding_cuda{}.pkl".format(self.model_cfg.TRIPLET_LOSS.DATASETS_NAME[0], input1.device.index))
            embedding = pkl.load(open(embedding_path, 'rb'))
        except:
            print(embedding_path, 'does not exist.')
            return 0, tb_dict
        last_input1, last_input2 = embedding['input1'], embedding['input2']
        triplet_loss_inter = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
        # l: last, c: current.
        if (len(input1) == 0) and (len(input2) == 0):
            return 0, tb_dict
        elif len(input1) == 0:
            dist_l1c2, dist_l2c2 = torch.cdist(last_input1, input2, p=2), torch.cdist(last_input2, input2, p=2)
            p_input2, n_input2 = last_input2[dist_l2c2.max(dim=0).indices], last_input1[dist_l1c2.min(dim=0).indices]
            inter_triplet_loss = triplet_loss_inter(input2, p_input2, n_input2)
        elif len(input2) == 0:
            dist_l1c1, dist_l2c1 = torch.cdist(last_input1, input1, p=2), torch.cdist(last_input2, input1, p=2)
            p_input1, n_input1 = last_input1[dist_l1c1.max(dim=0).indices], last_input2[dist_l2c1.min(dim=0).indices]
            inter_triplet_loss = triplet_loss_inter(input1, p_input1, n_input1)
        else:
            triplet_loss_c, triplet_loss_l = 0, 0
            if self.model_cfg.TRIPLET_LOSS.TO_CUR:
                dist_l1c1, dist_l1c2 = torch.cdist(last_input1, input1, p=2), torch.cdist(last_input1, input2, p=2)
                dist_l2c1, dist_l2c2 = torch.cdist(last_input2, input1, p=2), torch.cdist(last_input2, input2, p=2)
                p_input1, n_input1 = last_input1[dist_l1c1.max(dim=0).indices], last_input2[dist_l2c1.min(dim=0).indices]
                p_input2, n_input2 = last_input2[dist_l2c2.max(dim=0).indices], last_input1[dist_l1c2.min(dim=0).indices]
                triplet_loss_c = triplet_loss_inter(input1, p_input1, n_input1) \
                    + triplet_loss_inter(input2, p_input2, n_input2)
            if self.model_cfg.TRIPLET_LOSS.TO_LAST:
                p_last_input1, n_last_input1 = input1[dist_l1c1.max(dim=1).indices], input2[dist_l1c2.min(dim=1).indices]
                p_last_input2, n_last_input2 = input2[dist_l2c2.max(dim=1).indices], input1[dist_l2c1.min(dim=1).indices]
                triplet_loss_l = triplet_loss_inter(last_input1, p_last_input1, n_last_input1) \
                    + triplet_loss_inter(last_input2, p_last_input2, n_last_input2)
            inter_triplet_loss = triplet_loss_c + triplet_loss_l
        inter_triplet_loss = inter_triplet_loss * loss_cfgs.LOSS_WEIGHTS['inter_triplet_loss']
        tb_dict['inter_triplet_loss'] = inter_triplet_loss.item() if inter_triplet_loss != 0 else 0
        return inter_triplet_loss, tb_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        gt_boxes3d_class1 = forward_ret_dict['gt_of_rois'][...,-1].view(-1)[fg_mask]
        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError
        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)

        intra_triplet_loss, intra_tb_dict = self.get_intra_triplet_layer_loss(self.forward_ret_dict)
        # if intra_triplet_loss != 0:
        #     print(1212, intra_triplet_loss)
        rcnn_loss += intra_triplet_loss
        tb_dict.update(intra_tb_dict)

        inter_triplet_loss, inter_tb_dict = self.get_inter_triplet_layer_loss(self.forward_ret_dict)
        rcnn_loss += inter_triplet_loss
        tb_dict.update(inter_tb_dict)

        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
