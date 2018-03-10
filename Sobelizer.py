import cv2
import numpy as np
import matplotlib.pyplot as plt

class Sobelizer:
    sobelx = None
    sobely = None
    abs_sobelx = None
    abs_sobely = None
    scaled_sobelx = None
    scaled_sobely = None
    gradmag = None
    absgraddir = None

    def __init__(self, gray_image, sobel_kernel = 3):
        self.sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        self.sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        self.abs_sobelx = np.absolute(self.sobelx)
        self.abs_sobely = np.absolute(self.sobely)
        self.scaled_sobelx = np.uint8(255 * self.abs_sobelx / np.max(self.abs_sobelx))
        self.scaled_sobely = np.uint8(255 * self.abs_sobely / np.max(self.abs_sobely))
        self.gradmag = np.sqrt(self.sobelx ** 2 + self.sobely ** 2)
        self.gradmag = (self.gradmag / np.max(self.gradmag) * 255).astype(np.uint8)
        self.absgraddir = np.arctan2(self.abs_sobelx, self.abs_sobely)

    def threshold_x(self, thresh_min=0, thresh_max=255):
        binary_output = np.zeros_like(self.scaled_sobelx)
        binary_output[(self.scaled_sobelx >= thresh_min) & (self.scaled_sobelx <= thresh_max)] = 1
        return binary_output

    def threshold_y(self, thresh_min=0, thresh_max=255):
        binary_output = np.zeros_like(self.scaled_sobely)
        binary_output[(self.scaled_sobely >= thresh_min) & (self.scaled_sobelx <= thresh_max)] = 1
        return binary_output

    def mag_thresh(self, thresh_min=0, thresh_max=255):
        binary_output = np.zeros_like(self.gradmag)
        binary_output[(self.gradmag >= thresh_min) & (self.gradmag <= thresh_max)] = 1
        return binary_output

    def dir_threshold(self, thresh_min=0, thresh_max=255):
        abs_sobel = np.arctan2(self.abs_sobelx, self.abs_sobely)
        mask = np.zeros_like(abs_sobel)
        mask[(abs_sobel >= thresh_min) & (abs_sobel <= thresh_max)] = 1
        return mask

if __name__ == '__main__':

    image = cv2.imread("./test_images/test4.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelizer = Sobelizer(image, sobel_kernel=7)
    gradx = sobelizer.threshold_x(thresh_min=50, thresh_max=200)
    grady = sobelizer.threshold_y(thresh_min=50, thresh_max=200)
    # mag_binary = sobelizer.mag_thresh(thresh_min=50, thresh_max=200)
    # dir_binary = sobelizer.dir_threshold(thresh_min=0.7, thresh_max=1.3)
    # combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    plt.figure(1)
    plt.imshow(sobelizer.sobelx, cmap="gray")
    plt.figure(2)
    plt.imshow(sobelizer.abs_sobelx, cmap="gray")
    plt.figure(3)
    plt.imshow(sobelizer.scaled_sobelx, cmap="gray")

    plt.figure(4)
    plt.imshow(sobelizer.sobely, cmap="gray")
    plt.figure(5)
    plt.imshow(sobelizer.abs_sobely, cmap="gray")
    plt.figure(6)
    plt.imshow(sobelizer.scaled_sobely, cmap="gray")
    plt.show()


