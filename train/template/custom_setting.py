from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_setup
from detectron2.config import get_cfg
import detectron2.utils.comm as comm
from sparseinst import add_sparse_inst_config


def custom_arg_parser(arg_parser):
    arg_parser.add_argument("--train_dataset_name", type=str, default="train2017",
                            help="train_custom_dataset_names, use ':' to integrate names just like 'a:b' ")
    arg_parser.add_argument("--train_json_path",
                            type=str,
                            default="labels/annotations.json",
                            help="train_json_paths_as_same_order_as_train_datasets_names,  use ':' to integrate")
    arg_parser.add_argument("--train_img_dir",
                            type=str,
                            default="labels",
                            help="train_img_dirs_as_same_order_as_train_datasets_names, use ':' to integrate")

    return arg_parser


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    train_dataset_name = args.train_dataset_name
    train_json_path = args.train_json_path
    train_img_dir = args.train_img_dir

    # register custom coco train dataset
    register_coco_instances(train_dataset_name, {}, train_json_path, train_img_dir)
    DatasetCatalog.get(train_dataset_name)
    cfg.DATASETS.TRAIN = (train_dataset_name,)

    # register custom coco test dataset
    register_coco_instances("coco_my_val", {}, train_json_path, train_img_dir)
    DatasetCatalog.get("coco_my_val")
    cfg.DATASETS.TEST = ("coco_my_val",)

    # get number of training images classes
    classes_num = len(MetadataCatalog.get(train_dataset_name).thing_classes)
    # set classes number of ROI_HEADS
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = classes_num

    # set number of per batch of images for training to GPU
    cfg.SOLVER.IMS_PER_BATCH = 2
    # number of iterations for each epoch(200 is the number of training images)
    ITERS_IN_ONE_EPOCH = int(26 / cfg.SOLVER.IMS_PER_BATCH)
    
    if ITERS_IN_ONE_EPOCH * 50 < 2000:
        MAX_ITER = 2000
    else:
        MAX_ITER = ITERS_IN_ONE_EPOCH * 50
    # max number of iterations for training
    cfg.SOLVER.MAX_ITER = MAX_ITER - 1
    # after complete steps (iters),  learning rate start decrease
    cfg.SOLVER.STEPS = (cfg.SOLVER.MAX_ITER * 0.8, cfg.SOLVER.MAX_ITER * 0.9)
    # save model every epoch(name of model is the number of epoch)
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH * 100
    # Don`t evaluate on test set during training
    cfg.TEST.EVAL_PERIOD = 0
    # set FP16 to accelerate training
    # cfg.SOLVER.AMP.ENABLED = True

    cfg.INPUT.MIN_SIZE_TRAIN = (720, )
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT.MIN_SIZE_TEST = 720
    # Maximum size of the side of the image during testing
    cfg.INPUT.MAX_SIZE_TEST = 1280

    cfg.INPUT.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"
    
    # final model pth save path
    cfg.OUTPUT_DIR = "./result"

    # not common configs
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = classes_num

    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")
    return cfg
