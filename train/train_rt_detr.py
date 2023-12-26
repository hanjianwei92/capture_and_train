from pathlib import Path
from ultralytics import RTDETR


def train_rt_detr(dataset_path: str,
                  pretrained_model_path: str,
                  epochs: int = 50,
                  batch_size: int = 5,
                  img_size: int = 640,
                  workers: int = 10,
                  device: int | list[int] | str = 0,
                  out_path: str = "result"):
    """
    Train RT-DETR model
    args:
    :param dataset_path: 表示数据集的路径，需要包含classes.yaml、train文件夹、val文件夹，
                         train文件夹和val文件夹下分别包含images文件夹和labels文件夹
    :param pretrained_model_path: 表示预训练模型的路径，需要rt-detr-l.pth
    :param epochs: 表示所有样本训练的轮数, default is 50
    :param batch_size: 表示一次训练所选取的样本数目, default is 5
    :param img_size: 表示输入图片的尺寸, default is 640
    :param workers: 表示读取数据的线程数目, default is 10
    :param device: 表示使用的GPU编号，default is 0, 可以使用多个GPU，如[0, 1, 2], 或者使用CPU，如'cpu'
    :param out_path: 表示输出路径，default is 'result'
    :return: None
    """

    model = RTDETR('rtdetr-l.yaml').load(pretrained_model_path)  # build from YAML and transfer weights
    # Train the model on the COCO8 example dataset for 100 epochs
    model.train(
        data=str(Path(dataset_path) / 'classes.yaml'),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project=out_path,
        name='rt-detr',
        exist_ok=True
    )


if __name__ == "__main__":
    train_rt_detr(dataset_path='../capture/yolo_dataset/', pretrained_model_path='rtdetr-l.pt')

