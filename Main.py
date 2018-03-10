import cv2
import time

from FrameProcessor import FrameProcessor


def process_frames(input_folder, output_folder, l=None, f=0):
    images_path = FrameProcessor.load_frames(input_folder)
    processor = FrameProcessor()

    frame_no = f
    for path in images_path[f:l]:
        img = cv2.imread(path)
        img = processor.process_bgr(img)
        cv2.imwrite(output_folder + "/" + str(frame_no) + ".jpg", img)
        print(frame_no)
        frame_no = frame_no + 1


def process_video(input_vid, output_vid):
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip
    white_output = output_vid
    clip1 = VideoFileClip(input_vid)
    processor = FrameProcessor()

    def process(frame):
        time.sleep(0.02)
        return processor.process_rgb(frame)

    white_clip = clip1.fl_image(process)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    # process_frames("./video_frames/project_video/" ,"./video_frames/project_video_annotated/", 1)
    process_video("./project_video.mp4", './project_video_annotated.mp4')
