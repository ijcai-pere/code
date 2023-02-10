from functools import partial
import numpy as np
from . import augmentor_utils, database_sampler
from ...utils import common_utils, self_training_utils, box_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import torch

class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None, load_dir=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.augmentor_configs = augmentor_configs
        self.load_dir = load_dir
        self.data_augmentor_queue = []
        self.ps_box_frame = None
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def update_pkt(self):
        print('update_pkt')
        self.ps_box_frame = self.read_ps_labels()

    def read_ps_labels(self):
        try:
            pkt, l_ = self_training_utils.load_last_pseudo_label(self.load_dir)
            print('cur pkt dir', l_)
        except:
            print('Ps label does not exist. ')
            return None
        ps_labels = np.zeros(0)
        ps_boxes = np.zeros([0, 9])
        list_frame = []
        for i in pkt.keys():
            ps_labels = np.concatenate([ps_labels, pkt[i]['gt_boxes'][:, 7]])
            ps_boxes = np.concatenate([ps_boxes, pkt[i]['gt_boxes']])
            for j in range(len(pkt[i]['gt_boxes'])):
                list_frame.append(i)
        np_frame = np.array(list_frame)
        ps_boxes1 = ps_boxes[(ps_labels == 1)]
        np_frame1 = np_frame[(ps_labels == 1)]
        ps_boxes2 = ps_boxes[(ps_labels == 2)]
        np_frame2 = np_frame[(ps_labels == 2)]
        return ps_boxes1, np_frame1, ps_boxes2, np_frame2

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_sampling, config=config)
        gt_boxes, points, gt_boxes_mask = augmentor_utils.global_sampling(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            sample_ratio_range=config['WORLD_SAMPLE_RATIO'],
            prob=config['PROB']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes_mask'] = gt_boxes_mask
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def normalize_object_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size, config=config)
        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'], config['SIZE_RES']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def wt_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.wt_sampling, config=config)
        if self.ps_box_frame is None:
            return data_dict
        else:
            ps_boxes1, np_frame1, ps_boxes2, np_frame2 = self.ps_box_frame
        gt_boxes = data_dict['gt_boxes']
        gt_classes = data_dict['gt_classes']
        iou_scores = data_dict['gt_scores']
        index = np.array(range(len(gt_classes)))
        index_1 = index[gt_classes == -1]
        index_2 = index[gt_classes == -2]
        try:
            p1 = (iou_scores[index_1]-config.NEG_THRESH[0])/(config.POS_THRESH[0]-config.NEG_THRESH[0])
            p1_wt = p1 > np.random.rand(len(index_1))
            index_1_wt = index_1[p1_wt]
            data_dict['gt_classes'][index_1_wt] = 1

            p2 = (iou_scores[index_2]-config.NEG_THRESH[1])/(config.POS_THRESH[1]-config.NEG_THRESH[1])
            p2_wt = p2 > np.random.rand(len(index_2))
            index_2_wt = index_2[p2_wt]
            data_dict['gt_classes'][index_2_wt] = 2
            index_wt = np.concatenate([index_1_wt, index_2_wt])
        except:
            print(len(index_1), len(index_2), 'p1 or p2 does not exist.')
            return data_dict
        gt_mv_boxes1 = gt_boxes[index_1]
        gt_mv_boxes2 = gt_boxes[index_2]
        gt_mv_boxes = np.concatenate([gt_mv_boxes1, gt_mv_boxes2])
        points = data_dict['points']
        large_sampled_gt_boxes = box_utils.enlarge_box3d(gt_mv_boxes[:, 0:7], extra_width=(0, 0, 0))
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)

        index_ps1 = np.random.randint(len(ps_boxes1), size=len(index_1_wt))
        ps_boxes1 = ps_boxes1[index_ps1]  # high confidence boxes.
        np_frame1 = np_frame1[index_ps1]
        wt_boxes1 = gt_boxes[index_1_wt]

        index_ps2 = np.random.randint(len(ps_boxes2), size=len(index_2_wt))
        ps_boxes2 = ps_boxes2[index_ps2]  # high confidence boxes.
        np_frame2 = np_frame2[index_ps2]
        wt_boxes2 = gt_boxes[index_2_wt]

        np_frame = np.concatenate([np_frame1, np_frame2], axis=0)
        ps_boxes = np.concatenate([ps_boxes1, ps_boxes2], axis=0)
        wt_boxes = np.concatenate([wt_boxes1, wt_boxes2], axis=0)
        obj_points_list = []
        if config.DATA_TYPE == 'kitti':
            for idx, cur_ps_box, cur_wt_boxes, i in zip(np_frame, ps_boxes[:, :7], wt_boxes, range(len(wt_boxes2))):
                lidar_file = self.root_path / 'training' / 'velodyne' / ('%s.bin' % idx)
                assert lidar_file.exists()
                points_ps_cur = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points_ps_cur[:, 0:3]), torch.from_numpy(cur_ps_box.reshape(1, -1))).numpy()
                points_cur_ps_box = points_ps_cur[point_indices[0] > 0]
                if points_cur_ps_box.shape[0] > config.MIN_POINTS:
                    points_cur_ps_box[:, :3] -= cur_ps_box.reshape(1, -1)[0, :3]
                    points_cur_ps_box[:, :3] /= (cur_ps_box.reshape(1, -1)[0, 3:6]/cur_wt_boxes.reshape(1, -1)[0, 3:6])
                    points_cur_ps_box[:, :3] = points_cur_ps_box[:, :3] + cur_wt_boxes.reshape(1, -1)[0, :3]
                    obj_points_list.append(points_cur_ps_box)
                else:
                    data_dict['gt_classes'][index_wt][i] = -data_dict['gt_classes'][index_wt][i]
            if len(obj_points_list) == 0:
                return data_dict
            obj_points = np.concatenate(obj_points_list, axis=0)
        data_dict['points'] = np.concatenate([points, obj_points], axis=0)
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict

    def re_prepare(self, augmentor_configs=None, intensity=None):
        self.data_augmentor_queue = []

        if augmentor_configs is None:
            augmentor_configs = self.augmentor_configs

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # scale data augmentation intensity
            if intensity is not None:
                cur_cfg = self.adjust_augment_intensity(cur_cfg, intensity)
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def adjust_augment_intensity(self, config, intensity):
        adjust_map = {
            'random_object_scaling': 'SCALE_UNIFORM_NOISE',
            'random_object_rotation': 'ROT_UNIFORM_NOISE',
            'random_world_rotation': 'WORLD_ROT_ANGLE',
            'random_world_scaling': 'WORLD_SCALE_RANGE',
        }

        def cal_new_intensity(config, flag):
            origin_intensity_list = config.get(adjust_map[config.NAME])
            assert len(origin_intensity_list) == 2
            assert np.isclose(flag - origin_intensity_list[0], origin_intensity_list[1] - flag)
            
            noise = origin_intensity_list[1] - flag
            new_noise = noise * intensity
            new_intensity_list = [flag - new_noise, new_noise + flag]
            return new_intensity_list

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        if config.NAME in ["random_object_scaling", "random_world_scaling"]:
            new_intensity_list = cal_new_intensity(config, flag=1)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        elif config.NAME in ['random_object_rotation', 'random_world_rotation']:
            new_intensity_list = cal_new_intensity(config, flag=0)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        else:
            raise NotImplementedError
