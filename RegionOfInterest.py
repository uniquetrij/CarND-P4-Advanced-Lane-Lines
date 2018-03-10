import numpy as np

from ColorComponents import ColorComponents


class RegionOfInterest:
    margin = None
    rows = None
    cols = None

    def crop(self, image, margin=75):
        self.margin = margin
        self.rows = image.shape[0]
        self.cols = image.shape[1]
        x1 = int(self.cols / 2 - margin)
        x2 = int(self.cols / 2 + margin)
        image = image[:, x1:x2, :]
        return image

    def create_binary(self, image):
        components = ColorComponents(image)

        bgr_r = components.getComponent("bgr_r")
        bgr_g = components.getComponent("bgr_g")
        bgr_b = components.getComponent("bgr_b")
        y_mask = (bgr_g > 180) & (bgr_r > 180) & (bgr_b < 150)
        w_mask = (bgr_g > 150) & (bgr_r > 150) & (bgr_b > 200)
        yuv_u = 255 - components.getComponent("yuv_u")
        u_mask = yuv_u > 145

        binary = w_mask | y_mask | u_mask
        return binary

    def decrop(self, image):
        padding = np.zeros((self.rows, int(self.cols / 2 - self.margin)))
        image = np.hstack((image, padding))
        image = np.hstack((padding, image))
        image = np.uint8(image)
        return image

if __name__ == '__main__':
    import cv2
    # img = cv2.imread("./output_images/undistorted_straight_lines1_birdeye.jpg")
    # roi = RegionOfInterest()
    # img = roi.crop(img)
    # img = roi.create_binary(img)
    # img = roi.decrop(img)
    # cv2.imwrite("./output_images/undistorted_straight_lines1_birdeye_decropped.jpg",
    #             np.uint8(img*255))
    # import matplotlib.pyplot as plt
    # plt.imshow(img, cmap="gray")
    # plt.show()

