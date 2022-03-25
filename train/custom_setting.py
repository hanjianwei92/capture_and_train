#-*- coding:utf-8 -*-
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_setup
from adet.config import get_cfg
import detectron2.utils.comm as comm


def custom_arg_parser(arg_parser):
    arg_parser.add_argument("--train_dataset_names", type=str, default="train2017",
                            help="train_custom_dataset_names, use ':' to integrate names just like 'a:b' ")
    arg_parser.add_argument("--train_json_paths",
                            type=str,
                            default="dataset/coco/annotations/instance_train20171.json",
                            help="train_json_paths_as_same_order_as_train_datasets_names,  use ':' to integrate")
    arg_parser.add_argument("--train_img_dirs",
                            type=str,
                            default="dataset/coco/train2017",
                            help="train_img_dirs_as_same_order_as_train_datasets_names, use ':' to integrate")

    arg_parser.add_argument("--test_dataset_names", type=str, default="test2017",
                            help="test_custom_dataset_names, use ':' to integrate names just like 'a:b' ")
    arg_parser.add_argument("--test_json_paths",
                            type=str,
                            default="dataset/coco/annotations/instance_test20171.json",
                            help="test_json_paths_as_same_order_as_test_datasets_names,  use ':' to integrate")
    arg_parser.add_argument("--test_img_dirs",
                            type=str,
                            default="dataset/coco/test2017",
                            help="test_img_dirs_as_same_order_as_test_datasets_names, use ':' to integrate")

    return arg_parser


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    train_dataset_names = args.train_dataset_names.split(":")
    train_json_paths = args.train_json_paths.split(":")
    train_img_dirs = args.train_img_dirs.split(":")
    for name, json_path, img_dir in zip(train_dataset_names, train_json_paths, train_img_dirs):
        register_coco_instances(name, {}, json_path, img_dir)
        MetadataCatalog.get(name)
        DatasetCatalog.get(name)

    test_dataset_names = args.test_dataset_names.split(":")
    test_json_paths = args.test_json_paths.split(":")
    test_img_dirs = args.test_img_dirs.split(":")
    for name, json_path, img_dir in zip(test_dataset_names, test_json_paths, test_img_dirs):
        register_coco_instances(name, {}, json_path, img_dir)
        MetadataCatalog.get(name)
        DatasetCatalog.get(name)

    cfg.DATASETS.TRAIN = tuple(train_dataset_names)
    cfg.DATASETS.TEST = tuple(test_dataset_names)
    classes_num = max([len(MetadataCatalog.get(name).thing_classes)
                                           for name in train_dataset_names + test_dataset_names])
    cfg.MODEL.FCOS.NUM_CLASSES = classes_num
    default_setup(cfg, args)

    setup_logger().info("set cfg.MODEL.FCOS.NUM_CLASSES to {}".format(classes_num))
    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg