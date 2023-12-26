import json
import yaml
import base64
import cv2
import numpy as np
import shutil

from pathlib import Path
from sklearn.model_selection import train_test_split


def get_img_from_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
        image_b64 = data["imageData"]

        image_data = base64.b64decode(image_b64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(Path(json_path).with_suffix(".png")), image)


def get_mask_from_polygon_points(points: list, image_shape: tuple):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def get_bbox_from_polygon_points(points: list):
    x_min = min([x[0] for x in points])
    x_max = max([x[0] for x in points])
    y_min = min([x[1] for x in points])
    y_max = max([x[1] for x in points])

    return x_min, y_min, x_max, y_max


def convert_anylabeling_format_to_yolo(input_path: str, out_path: str = None):
    """
    Convert anylabeling format to yolo format
    :param input_path: anylabeling format path
    :param out_path: yolo format path
    :return: None
    """
    input_path = Path(input_path)
    if out_path is None:
        out_path = Path(input_path).parent / "yolo_dataset"
    else:
        out_path = Path(out_path)

    train_image_path = out_path / "train" / "images"
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_label_path = out_path / "train" / "labels"
    train_label_path.mkdir(parents=True, exist_ok=True)

    val_image_path = out_path / "val" / "images"
    val_image_path.mkdir(parents=True, exist_ok=True)
    val_label_path = out_path / "val" / "labels"
    val_label_path.mkdir(parents=True, exist_ok=True)

    yaml_path = out_path / "classes.yaml"

    class_name_to_id = {}
    class_id = 0
    class_name_to_id["_background_"] = class_id

    file_stems = [x.stem for x in input_path.glob("*.json")]
    _, val_filestems = train_test_split(file_stems, test_size=0.1, random_state=50)

    for filename in input_path.glob("*.json"):
        with open(filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        image_width = json_data["imageWidth"]
        image_height = json_data["imageHeight"]

        if filename.stem in val_filestems:
            target_label = (val_label_path / filename.name).with_suffix(".txt")
        else:
            target_label = (train_label_path / filename.name).with_suffix(".txt")

        with open(target_label, "w") as f:
            for s_data in json_data["shapes"]:
                label = s_data["label"]
                xyxy = get_bbox_from_polygon_points(s_data['points'])

                if label not in class_name_to_id:
                    class_id += 1
                    class_name_to_id[label] = class_id

                current_class_id = class_name_to_id[label]

                x_center = (xyxy[0] + xyxy[2]) / (2 * image_width)
                y_center = (xyxy[1] + xyxy[3]) / (2 * image_height)
                width = (xyxy[2] - xyxy[0]) / image_width
                height = (xyxy[3] - xyxy[1]) / image_height

                f.write(f"{current_class_id} {x_center} {y_center} {width} {height}\n")

    for img in input_path.glob("*.jpg"):
        if img.stem in val_filestems:
            # img.rename(val_image_path / img.name)
            shutil.copy(img, str(val_image_path / img.name))
        else:
            # img.rename(train_image_path / img.name)
            shutil.copy(img, str(train_image_path / img.name))
    for img in input_path.glob("*.png"):
        if img.stem in val_filestems:
            # img.rename(val_image_path / img.name)
            shutil.copy(img, str(val_image_path / img.name))
        else:
            # img.rename(train_image_path / img.name)
            shutil.copy(img, str(train_image_path / img.name))

    with open(yaml_path, "w") as f:
        yaml_data = {
            "path": str(out_path.resolve()),
            "train": "train/images",
            "val": "val/images",
            "names": {v: k for k, v in class_name_to_id.items()}
        }
        yaml.dump(yaml_data, f)


if __name__ == "__main__":
    convert_anylabeling_format_to_yolo("../capture/img")
