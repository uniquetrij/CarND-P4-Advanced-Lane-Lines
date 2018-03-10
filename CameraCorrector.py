import glob
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt


class CameraCorrector:

    ret = None
    mtx = None
    dist = None

    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    @classmethod
    def from_pickle(cls, pickle_file):
        with open(pickle_file, mode='rb') as f:
            camera_corrector = pickle.load(f)
            return CameraCorrector(camera_corrector['mtx'], camera_corrector['dist'])

    @classmethod
    def from_images(cls, image_paths, nx, ny, pickle_path = None):

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in image_paths:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)

        if pickle_path is not None:
            camera_corrector = {"mtx": mtx, "dist": dist}
            pickle.dump(camera_corrector, open(pickle_path, "wb"))
        return CameraCorrector(mtx, dist)

    def correct(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)


if __name__ == '__main__':

    # import sys
    # import os
    # sys.path.append(os.getcwd())
    # corrector = cam_corrector.from_images(glob.glob('./camera_corrector/calibration*.jpg'), 9, 6, "./camera_corrector/camera_corrector.p")
    corrector = CameraCorrector.from_pickle('./camera_corrector/camera_corrector.p')
    a = plt.imshow(corrector.correct(cv2.imread('./lane_aerializer/straight_lines1.jpg')))
    cv2.imwrite("./output_images/undistorted_straight_lines1.jpg",a.get_array())





