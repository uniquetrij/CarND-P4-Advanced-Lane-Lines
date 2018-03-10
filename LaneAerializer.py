import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np


class LaneAerializer:
    mtx = None
    trapiz = None
    rect = None
    mtxInv = None

    def __init__(self, trapiz, rect):
        self.trapiz = trapiz
        self.rect = rect
        self.mtx = cv2.getPerspectiveTransform(trapiz, rect)
        self.mtxInv = cv2.getPerspectiveTransform(rect, trapiz)

    @classmethod
    def from_pickle(cls, pickle_file):
        with open(pickle_file, mode='rb') as f:
            pers_trans = pickle.load(f)
            return LaneAerializer(pers_trans['trapiz'], pers_trans['rect'])

    @classmethod
    def from_image(cls, image_path, pickle_path=None, grid_interval=25, grid_color=(200, 100, 200)):
        points = []
        current = [0, 0]
        image = cv2.imread(image_path)
        width = image.shape[1]
        height = image.shape[0]
        img = image.copy()

        c_x = int(width / 2)
        c_y = int(height / 2)

        for i in range(0, c_x + 1, grid_interval):
            cv2.line(img, (i, 0), (i, height), grid_color, 1)
            cv2.line(img, (width - i, 0), (width - i, height), grid_color, 1)

        for i in range(0, c_y + 1, grid_interval):
            cv2.line(img, (0, i), (width, i), grid_color, 1)
            cv2.line(img, (0, height - i), (width, height - i), grid_color, 1)

        def select_point(event, x, y, flags, param):
            current[0] = x
            current[1] = y
            if event == cv2.EVENT_LBUTTONDBLCLK:
                points.append([x, y])

        cv2.namedWindow('image')
        cv2.resizeWindow('image', 200, 200)
        cv2.setMouseCallback('image', select_point)
        cv2.moveWindow('image', 0, 0)

        while True:
            temp_img = img.copy()
            cv2.putText(temp_img, str(current), (current[0] + 20, current[1]), cv2.FONT_HERSHEY_PLAIN, 0.5,
                        (255, 255, 255), 1)
            for point in points:
                cv2.circle(temp_img, (point[0], point[1]), 1, (255, 0, 0), -1)
            cv2.imshow('image', temp_img)
            k = cv2.waitKey(20) & 0xFF
            if k == 8:
                try:
                    points.pop()
                except:
                    pass
            if k == 27:
                break

        trapiz = np.float32(np.array(points.copy()))
        mark = 0.47 * width

        rect = np.float32([(mark, 0),
                           (width - mark, 0),
                           (width - mark, height),
                           (mark, height)])

        temp_img = image.copy()

        cv2.polylines(temp_img, [np.int32(rect)], 1, (0, 0, 255), 3)
        cv2.polylines(temp_img, [np.int32(trapiz)], 1, (0, 255, 0), 3)
        cv2.imshow('image', temp_img)
        cv2.waitKey(200000)
        cv2.destroyAllWindows()
        cv2.imwrite("./output_images/undistorted_straight_lines1_perspective_overlay.jpg", temp_img)
        if len(trapiz) == 4 & len(rect) == 4:
            if pickle_path is not None:
                pers_trans = {}
                pers_trans["trapiz"] = np.array(trapiz)
                pers_trans["rect"] = np.array(rect)
                pickle.dump(pers_trans, open(pickle_path, "wb"))

            return LaneAerializer(trapiz, rect)
        else:
            raise RuntimeError

    def aerialize(self, image):
        return cv2.warpPerspective(image, self.mtx, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def deaerialize(self, image):
        return cv2.warpPerspective(image, self.mtxInv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    lane_aerializer = LaneAerializer.from_image('./output_images/undistorted_straight_lines1.jpg', "./lane_aerializer/lane_aerializer.p")
    # lane_aerializer = LaneAerializer.from_pickle('./lane_aerializer/lane_aerializer.p')

    # image = cv2.imread("./lane_aerializer/straight_lines2.jpg")
    # trapiz = np.float32([(575, 464),
    #                      (707, 464),
    #                      (258, 682),
    #                      (1049, 682)])
    # mark = 600
    # rect = np.float32([(mark, 0),
    #                    (image.shape[1] - mark, 0),
    #                    (mark, image.shape[0]),
    #                    (image.shape[1] - mark, image.shape[0])])
    # lane_aerializer = {"trapiz": np.float32(np.array(trapiz)), "rect": np.float32(np.array(rect))}
    # pickle.dump(lane_aerializer, open("./lane_aerializer/lane_aerializer.p", "wb"))
    # lane_aerializer = LaneAerializer(trapiz, rect)
    cv2.imwrite("./output_images/undistorted_straight_lines1_birdeye.jpg",lane_aerializer.aerialize(cv2.imread('./lane_aerializer/straight_lines1.jpg')))

    # plt.imshow(lane_aerializer.aerialize(cv2.imread('./lane_aerializer/straight_lines1.jpg')))
    # plt.show()
