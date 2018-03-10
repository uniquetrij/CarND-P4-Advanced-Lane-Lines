import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ColorComponents:
    bgr = None
    gray = None
    rgb = None
    hls = None
    hsv = None
    lab = None
    luv = None
    yuv = None
    ycrcb = None
    xyz = None

    channels = {}
    spaces = {}

    def __init__(self, image_bgr):
        self.bgr = image_bgr
        self.channels = {
            "gray": self.get_gray,
            "bgr_b": self.get_bgr_b, "bgr_g": self.get_bgr_g, "bgr_r": self.get_bgr_r,
            "rgb_r": self.get_rgb_r, "rgb_g": self.get_rgb_g, "rgb_b": self.get_rgb_b,
            "hls_h": self.get_hls_h, "hls_l": self.get_hls_l, "hls_s": self.get_hls_s,
            "hsv_h": self.get_hsv_h, "hsv_s": self.get_hsv_s, "hsv_v": self.get_hsv_v,
            "lab_l": self.get_lab_l, "lab_a": self.get_lab_a, "lab_b": self.get_lab_b,
            "luv_l": self.get_luv_l, "luv_u": self.get_luv_u, "luv_v": self.get_luv_v,
            "yuv_y": self.get_yuv_y, "yuv_u": self.get_yuv_u, "yuv_v": self.get_yuv_v,
            "ycrcb_y": self.get_ycrcb_y, "ycrcb_cr": self.get_ycrcb_cr, "ycrcb_cb": self.get_ycrcb_cb,
            "xyz_x": self.get_xyz_x, "xyz_y": self.get_xyz_y, "xyz_z": self.get_xyz_z,
        }
        self.spaces = {
            "bgr": self.get_bgr,
            "rgb": self.get_rgb,
            "hls": self.get_hls,
            "hsv": self.get_hsv,
            "lab": self.get_lab,
            "luv": self.get_luv,
            "yuv": self.get_yuv,
            "ycrcb": self.get_ycrcb,
            "xyz": self.get_xyz
        }

    @classmethod
    def from_file(cls, image_path):
        return ColorComponents(cv2.imread(image_path))

    @classmethod
    def from_channels(cls, ch1, ch2, ch3):
        img = ch1
        img = np.dstack((img, ch2))
        img = np.dstack((img, ch3))
        return ColorComponents(img)

    def get_gray(self):
        if self.gray is None:
            self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self.gray

    def get_bgr(self):
        return self.bgr

    def get_bgr_b(self):
        return self.get_bgr()[:, :, 0]

    def get_bgr_g(self):
        return self.get_bgr()[:, :, 1]

    def get_bgr_r(self):
        return self.get_bgr()[:, :, 2]

    def get_rgb(self):
        if self.rgb is None:
            self.rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        return self.rgb

    def get_rgb_r(self):
        return self.get_rgb()[:, :, 0]

    def get_rgb_g(self):
        return self.get_rgb()[:, :, 1]

    def get_rgb_b(self):
        return self.get_rgb()[:, :, 2]

    def get_hls(self):
        if self.hls is None:
            self.hls = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HLS)
        return self.hls

    def get_hls_h(self):
        return self.get_hls()[:, :, 0]

    def get_hls_l(self):
        return self.get_hls()[:, :, 1]

    def get_hls_s(self):
        return self.get_hls()[:, :, 2]

    def get_hsv(self):
        if self.hsv is None:
            self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        return self.hsv

    def get_hsv_h(self):
        return self.get_hsv()[:, :, 0]

    def get_hsv_s(self):
        return self.get_hsv()[:, :, 1]

    def get_hsv_v(self):
        return self.get_hsv()[:, :, 2]

    def get_lab(self):
        if self.lab is None:
            self.lab = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB)
        return self.lab

    def get_lab_l(self):
        return self.get_lab()[:, :, 0]

    def get_lab_a(self):
        return self.get_lab()[:, :, 1]

    def get_lab_b(self):
        return self.get_lab()[:, :, 2]

    def get_luv(self):
        if self.luv is None:
            self.luv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LUV)
        return self.luv

    def get_luv_l(self):
        return self.get_luv()[:, :, 0]

    def get_luv_u(self):
        return self.get_luv()[:, :, 1]

    def get_luv_v(self):
        return self.get_luv()[:, :, 2]

    def get_yuv(self):
        if self.yuv is None:
            self.yuv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2YUV)
        return self.yuv

    def get_yuv_y(self):
        return self.get_yuv()[:, :, 0]

    def get_yuv_u(self):
        return self.get_yuv()[:, :, 1]

    def get_yuv_v(self):
        return self.get_yuv()[:, :, 2]

    def get_ycrcb(self):
        if self.ycrcb is None:
            self.ycrcb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2YCR_CB)
        return self.ycrcb

    def get_ycrcb_y(self):
        return self.get_ycrcb()[:, :, 0]

    def get_ycrcb_cr(self):
        return self.get_ycrcb()[:, :, 1]

    def get_ycrcb_cb(self):
        return self.get_ycrcb()[:, :, 2]

    def get_xyz(self):
        if self.xyz is None:
            self.xyz = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2XYZ)
        return self.xyz

    def get_xyz_x(self):
        return self.get_xyz()[:, :, 0]

    def get_xyz_y(self):
        return self.get_xyz()[:, :, 1]

    def get_xyz_z(self):
        return self.get_xyz()[:, :, 2]

    def getComponent(self, channel):
        return self.channels[channel]()

    def getSpace(self, space):
        return self.spaces[space]()

if __name__ == '__main__':
    images_path = glob.glob("./test_images/*.jpg")
    image = cv2.imread(images_path[5])
    cs = ColorComponents(image)
    # cs.show_each()
    plt.imshow(cv2.cvtColor(cs.getSpace("bgr"), cv2.COLOR_BGR2RGB))
    plt.show()


