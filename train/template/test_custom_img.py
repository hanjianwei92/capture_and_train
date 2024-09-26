import cv2
import time
import torch
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from adet.config import get_cfg as adet_get_cfg
from sparseinst import add_sparse_inst_config
from dcamera.realsense import SelfClbRealsense_K_D


class Detectron2Detector:
    def __init__(self,
                 model_weight: str,
                 config_file: str,
                 confidence_threshold: float = None,
                 config_mothod=get_cfg):
        super().__init__()
        cfg = config_mothod()
        add_sparse_inst_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_weight
        if confidence_threshold is not None:
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            
        # build labels text, it isn`t necessary for predictor to predict image 
        register_coco_instances("coco_my_val", {}, "labels/annotations.json", "labels")
        DatasetCatalog.get("coco_my_val")
        cfg.DATASETS.TEST = ("coco_my_val",)
        self.metadata = MetadataCatalog.get("coco_my_val")
        # self.metadata = MetadataCatalog.get("__unused")
        
        cfg.freeze()

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE

    def detect(self, frame):
        image = frame
        time_mark1 = time.time()
        predictions = self.predictor(image)
        print("predict one image = {}\n".format(time.time() - time_mark1))
        instances = predictions["instances"]
        # image = image[:, :, ::-1]
        # visualizer = Visualizer(image, None, instance_mode=self.instance_mode)
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        instances = instances.to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return vis_output.get_image().copy()


if __name__ == "__main__":
    ysg = Detectron2Detector(
                             model_weight="result/model_final.pth",
                             config_file="result/config.yaml",
                             config_mothod=get_cfg,
                             # model_weight="Blendmask/model_final.pth",
                             # config_file="Blendmask/config.yaml",
                             # config_mothod=adet_get_cfg,
                             confidence_threshold=0.5,
                             )
    cam = SelfClbRealsense_K_D()
    # cnt = 0
    while True:
        # color_image = cv2.imread(f"labels/JPEGImages/{cnt}.jpg")
        # cnt += 1
        color_image = cam.get_frame()[0]
        img_show = ysg.detect(color_image[:, :, ::-1])
        cv2.imshow("object_points", img_show)
        key = cv2.waitKey(1)
        if key == 27:
            break
