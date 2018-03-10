import glob
import matplotlib.pyplot as plt
import imageio
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

imageio.plugins.ffmpeg.download()
import cv2
import re

from CameraCorrector import CameraCorrector
from ColorComponents import ColorComponents
from LaneAerializer import LaneAerializer
from LaneIdentifier import LaneIdentifier
from RegionOfInterest import RegionOfInterest

class FrameProcessor:
    corrector = None
    aerializer = None
    laneidentifier = None
    roi = None

    curvatures = []

    def __init__(self, corrector=CameraCorrector.from_pickle('./camera_corrector/camera_corrector.p'),
                 aerializer=LaneAerializer.from_pickle('./lane_aerializer/lane_aerializer.p')
                 , laneidentifier=LaneIdentifier(25)):
        self.corrector = corrector
        self.laneidentifier = laneidentifier
        self.aerializer = aerializer
        self.roi = RegionOfInterest()

    def annotation(self, image):
        viz, left_fit, right_fit, ploty, left_fitx, right_fitx = self.laneidentifier.search(image)

        warp_zero = np.zeros_like(image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = self.aerializer.deaerialize(color_warp)
        return newwarp

    def annotate(self, annotation, original):
        # Combine the result with the original image
        result = cv2.addWeighted(original, 1, annotation, 0.3, 0)
        return result

    def write_curvature(self, image):

        curvature_l, curvature_r, off_position = self.laneidentifier.get_curvature()
        current = np.min([curvature_l, curvature_r])
        self.curvatures.append(current)
        # mean = np.mode(self.curvatures)
        # mean = stats.mode(self.curvatures)
        mean = np.median(self.curvatures)

        current = " current curvature (km) : %07.3f" % (current / 1000)
        mean = "    mean curvature (km) : %07.3f" % (mean / 1000)
        off_position = "        off center (cm) : %07.2f" % abs(off_position*100)


        # print(current, mean)

        cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        draw = ImageDraw.Draw(pil_im)
        # use a truetype font
        font = ImageFont.truetype("./Consolas.ttf", 16)

        # Draw the text
        draw.text((20, 20), "(-)ve & (+)ve values indicate 'left' & 'right' respectively", font=font)
        draw.text((20, 35), current, font=font)
        draw.text((20, 50), mean, font=font)
        draw.text((20, 64), off_position, font=font)

        # Get back the image to OpenCV
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        return cv2_im_processed

    def pipeline(self, image):
        image = self.corrector.correct(image)
        original = image.copy()
        image = self.aerializer.aerialize(image)
        image = self.roi.crop(image)
        image = self.roi.create_binary(image)
        image = self.roi.decrop(image)
        image = self.annotation(image)
        image = self.annotate(image, original)
        image = self.write_curvature(image)
        return image



    def process_bgr(self, frame_bgr):
        return self.pipeline(frame_bgr)

    def process_rgb(self, frame_rgb):
        return cv2.cvtColor(self.process_bgr(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)),cv2.COLOR_BGR2RGB)

    @classmethod
    def save_frames(cls, video_file, save_path):
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            if (success):
                print('Reading frame: ' + str(count + 1), success)
                cv2.imwrite(save_path + "/%d.jpg" % count, image)
                count += 1

    @classmethod
    def sorted_aphanumeric(cls, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    @classmethod
    def load_frames(cls, load_path):
        return cls.sorted_aphanumeric(glob.glob(load_path + "/*"))


# if __name__ == '__main__':
#     images_path = FrameProcessor.load_frames("./video_frames/project_video")
#
#     processor = FrameProcessor()
#
#
#     i = 0
#     j = None
#     for path in images_path[i:j]:
#         # try:
#         img = cv2.imread(path)
#         img = processor.process_bgr(img)
#         cv2.imwrite("./video_frames/project_video_annotated/" + str(i) + ".jpg", img)
#         print(i)
#         i = i + 1


#
#
#
if __name__ == '__main__':
    # FrameProcessor.save_frames("./project_video.mp4", "./video_frames/project_video")
    # FrameProcessor.save_frames("./challenge_video.mp4", "./video_frames/challenge_video")
    # FrameProcessor.save_frames("./harder_challenge_video.mp4", "./video_frames/harder_challenge_video")

    # print(len(load("./video_frames/project_video")))

    processor = FrameProcessor()
    img = cv2.imread("./lane_aerializer/straight_lines1.jpg")
    img = processor.process_bgr(img)
    cv2.imwrite("./output_images/final_result.jpg", img)

    # images_path = FrameProcessor.load_frames("./video_frames/project_video")
    # processor = FrameProcessor()
    # i = 0
    # j = None
    # for path in images_path[i:j]:
    #     img = cv2.imread(path)
    #     img = processor.process_bgr(img)
    #     cv2.imwrite("./video_frames/project_video_annotated/" + str(i) + ".jpg", img)
    #     print(i)
    #     i = i + 1
    #     time.sleep(0.02)
