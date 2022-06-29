from pathlib import Path
import cv2
import numpy as np
from dcamera.mechmind import Mechmind
import argparse


class CaptureImage:
    def __init__(self,
                 dcamera,
                 project_root_path: str):

        self.rgb_path = Path(project_root_path) / "img"
        self.rgb_path.mkdir(exist_ok=True, parents=True)
        self.rs = dcamera

    def get_rgb(self):
        rgb = self.rs.get_frame()[0]
        return rgb[:, :, ::-1]

    def save_rgb(self, cnt, img):
        # img2 = rotation(img, 180)
        # depth2 = rotation(depth, 180)
        cv2.imwrite(str(self.rgb_path / f"{cnt}.png"), img)

    @staticmethod
    def rotation(img, theta):
        if len(img.shape) == 3:
            rows, cols, channels = img.shape
        else:
            rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst


if __name__ == '__main__':
    mechmind = Mechmind(flip_nums=1)
    project_path = Path(__file__).parent
    capturer = CaptureImage(dcamera=mechmind, project_root_path=str(project_path))
    cnt = len([img for img in capturer.rgb_path.glob('*.png')])
    print("Press space to capture image!")
    while True:
        img = capturer.get_rgb()
        cv2.imshow('rgb', img)
        key = cv2.waitKey(1)

        if key == 27:
            print("Done!")
            break

        if key == ord(" "):
            capturer.save_rgb(cnt, img)
            print(f'save img {cnt}')
            cnt += 1
