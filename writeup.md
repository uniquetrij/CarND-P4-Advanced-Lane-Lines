
## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1.0]: ./camera_corrector/calibration1.jpg "Original"
[image1.1]: ./output_images/undistorted_calibration1.jpg "Undistorted"
[image2.0]: ./lane_aerializer/test1.jpg "Road Transformed"
[image2.1]: ./output_images/undistorted_test1.jpg "Undistorted Road Transformed"
[image3.0]: ./output_images/interactive_window.png "Interactive Window"
[image3.1]: ./output_images/undistorted_straight_lines1_perspective_overlay.jpg "Transformation Rects"
[image3.2]: ./output_images/undistorted_straight_lines1_birdeye.jpg "Bird-View Transform"

[image4.0]: ./output_images/undistorted_straight_lines1_birdeye_cropped.jpg "Cropped"
[image4.1]: ./output_images/undistorted_straight_lines1_birdeye_cropped_binary_y.jpg "y_mask"
[image4.2]: ./output_images/undistorted_straight_lines1_birdeye_cropped_binary_w.jpg "w_mask"
[image4.3]: ./output_images/undistorted_straight_lines1_birdeye_cropped_binary_u.jpg "u_mask"
[image4.4]: ./output_images/undistorted_straight_lines1_birdeye_cropped_binary_bin.jpg "binary"
[image4.5]: ./output_images/undistorted_straight_lines1_birdeye_decropped.jpg "decropped"

[image5.0]: ./output_images/histogram.jpg "Histogram"
[image5.1]: ./output_images/lane_curves_straight_lines1.jpg "Lane Curves"

[image5.2]:  ./lane_aerializer/test5.jpg "Original Image"
[image5.3]: ./output_images/lane_curves_test5.jpg "Lane Curves"

[image6.0]: ./output_images/final_result.jpg "Final Result"



[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.   is a template writeup for this project you can use as a guide and a starting point.  

You're reading it [Here](https://github.com/uniquetrij/CarND-P4-Advanced-Lane-Lines/blob/master/writeup.md)!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera distortions can be corrected using an instance of the class `CameraCorrector` in file named `CameraCorrector.py`. An instance, say `corrector` of `CameraCorrector` can be obtained in either of the following ways:
1. by passing the camera-calibration-matrix and distortion-coeffisients to the class initializer (code lines 14 through 16).

2. calculating the camera-calibration-matrix and distortion-coeffisients from chess-board images captured by camera in question. If a pickle file path is provided, it automatically  saves the calculated camera-calibration-matrix and distortion-coeffisients into the pickle file (code lines 24 through 53).
    
3. loading the camera-calibration-matrix and distortion-coeffisients from a pickle file previously saved (code lines 18 through 22).

To calculate the camera-matrix and distortion-coeffisients mentioned above, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

Once the `corrector` is obtained, I then obtain the corrected (undistorted) image by passing the image through `corrector.correct()` method (code lines 55 through 56) which internally uses `cv2.undistort()` function. The test result obtained is as follows:

|Original Image         |Undistorted Image    |
|:---------------------:|:-------------------:| 
|![alt text][image1.0]  |![alt text][image1.1]|


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

My first step in the pipeline is to correct the image of its camera distortions. To demonstrate this step, I will describe how I apply the distortion correction to one of the test images of the road like this one:

|Original Image         |Undistorted Image    |
|:---------------------:|:-------------------:| 
|![alt text][image2.0]  |![alt text][image2.1]|

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Instead of performing color transforms, gradients or other methods to create a thresholded binary image before perspective transform, I decide to perform the perspective transform right after the camera correction. Since, in essence, the perspective transform warps the image of the lane-lines into a bird-eye or aerial view, I create a class that does this job and name it `LaneAerializer` that can be found in file `LaneAerializer.py`. A `LaneAerializer` object may be used to "aerialize" (i.e. obtain the bird-view transformation of) the lane lines in the image, or "deaerialize" (i.e. perform the inverse perspective transform of) a bird-view image.

An object, say `aerializer`, of `LaneAerializer` may be instantiated in the following 3 ways:
1. by passing the source and destination quadrilateral vertices to the initializer of the class. The method then internally calculates the transformation matrix as well as the inverse transformation matrix for the required perspective transformation (code lines 13 through 17).

2. from a perspective view image of straight and parallel lane lines loaded in an interactive window where the user can click to select the vertices of the perspective trapezium that would otherwise be rectangular in an aerial view. If a pickle file path is provided, it saves the selected vertices so that it may be reused (code lines 25 through 96). 

3. loading the vertices from a pickle file previously saved (code lines 19 through 23).

To obtain the perspective transformation matrix, I first load the undistorted test image (that captures the perspective of straight and parallel lane lines) into an interactive window where I can click and select the pixels of a trapezium that defines the perspective of the lane lines. The example is as follows.

|Interactive Window     |Selection Areas      |
|:---------------------:|:-------------------:| 
|![alt text][image3.0]  |![alt text][image3.1]|

The image on the left shows the interactive window where the blue cross hair marks represent a few of the selected points. The image on the right shows the result. The green trapezium is the selected lane perspective that will be warped into the rectangular region marked in red. Notice that I kept the width of the rectangular region lesser than the minimum width of the trapezium. This prevents blurring of the lane lines during transformation due to interpolation. This results in an image with very crisp lane lines in the region of interest without much distortion (code lines 72 through 78).

The resulting source (trapezium) and destination (rectangle) points are as follows:

| Source (Trapezium-Green)  | Destination (Rectangle-Red)   | 
|:-------------------------:|:-----------------------------:| 
| 590, 452                  | 601.6, 0                      | 
| 688, 452                  | 678.4, 0                      |
| 1067, 688                 | 678.4, 720                    |
| 239, 688                  | 601.6, 720                    |

Once the `aerializer` is ontained, it may be used to perform perspective transform or the inverse of it using the methods `aerializer.aerialize()` and `aerializer.deaerialize()` respectively (code lines 98 through 102).
Following is the resulting transformed bird-view image
:

|Bird-View Image        |
|:---------------------:|
|![alt text][image3.2]  |

The above bird-view image is very interesting. The lane lines, on which the car has to move, are very crisp; but regions to the left and right of the lanes are mostly blurred out. 


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Next, I moved on to processing the region on interest, which is the central part of the region in the horizontal direction, where the lane lines are. 

First, most of the left anf right regions of the image are blurred out. Hence I decide to crop-off the left and right regions of the image and keep only the central lane lines with some buffer margin region (to be able to detect curves in case the lane turns to the left or right). I decide to keep only the region between +/-75 pixels from the center of the image in the horizontal direction and crop off the restof the region. This will result in a better binary processing with leser noise. 

I create a class `RegionOfInterest` (file named `RegionOfInterest.py`). It contains a function named `crop()` that I use for cropping the image (code lines 11 through 18). The resulting image is as follows:

|Cropped Bird-View Image        |
|:-----------------------------:|
|![alt text][image4.0]          |

Second, Its time to create the binary image from the cropped image above. For this, I tried out various techniques including using sobel operator to obtain straight line information from the image, finding magnitude, gradient and direction, using different color spaces and combinations of all these. For this, I created two dedicated classes; 1. Sobelizer (file `Sobelizer.py`) and 2. `ColorComponents` (file `ColorComponents.py`) to experiment with these operations. Ultimately I found out that using color spaces (especially BGR and YUV) alone was enough to detect the lane lines properly. Thresholding the RGB channels gave a nice representation of the white line lines, while thresholding the U channel of YUV gave a very prominent representation of the yellow lane lines.

My BGR thresholding is as follows (`RegionOfInterest.py` code lines 23 through 27) where I extract the Yellow color and the White color From BGR channels and the U channel of YUV (inverted) and then combine them into a binary image:

```python

        bgr_r = components.getComponent("bgr_r")
        bgr_g = components.getComponent("bgr_g")
        bgr_b = components.getComponent("bgr_b")
        y_mask = (bgr_g > 180) & (bgr_r > 180) & (bgr_b < 150)
        w_mask = (bgr_g > 150) & (bgr_r > 150) & (bgr_b > 200)
        yuv_u = 255 - components.getComponent("yuv_u")
        u_mask = yuv_u > 145
        binary = w_mask | y_mask | u_mask
```

|y_mask (yellow-left)  |w_mask (white-right)  |u_mask (yellow-left)  |binary (combined)     |
|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|![alt text][image4.1] |![alt text][image4.2] |![alt text][image4.3] |![alt text][image4.4] |

This combination worked out really well, and hence I went forward with this.

After this operation I expanded the image to the original size by padding zeros to it using the `decrop()` function of `RegionOfInterest`. The result is: 

|Decropped Binary  Image        |
|:-----------------------------:|
|![alt text][image4.5]          |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After obtaining the full binary image, I proceed to fit my lane lines with a 2nd order polynomial. To do so I used the code provided in the Udacity classroom material to obtain the polynomial. I created a class `LaneIdentifier` (file `LaneIdentifier.py`) that helps in generating the polynomial function and to obtain a list of coordinates of a polygon representing the lane region between the two lane lines. This may be done using the `search()` method of the class. When the `search()` is called for the first time, it internally makes a call to the `search_first()` (code lines 58 through 149) method where a histogram is generated to find out the base of the lane lines (code line 61). 

|Lane Position Histogram        |
|:-----------------------------:|
|![alt text][image5.0]          |

In the second and subsequent calls to `search()`, it internally calls `search_next()` (code lines 151 through 213) where the base position of the lane lines are inferred from the previous calculations. Finally, the search method returns the list of coordinates of the lane curves defined by the 2nd order polynomial function as follows:

|Polynomial Plot                |
|:-----------------------------:|
|![alt text][image5.1]          |

Here is another example, for a curved road:

|Original Image             |
|:-------------------------:|
|![alt text][image5.2]      |

|Polynomial Curve           |
|:-------------------------:|
|![alt text][image5.3]      |

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

My `LaneIdentifier` class also contains a method `get_curvature()` that calculates the curvature of the left and right lanes and also the offset of the car from the center of the lane and returns them (code lines 19 through 44). Here again, I directly used the code provided in the Udacity classroom to calculate the curvature. Only parameter I had to adjust here was the pixel to meters conversion in the horizontal direction, since I had contracted the image during the perspective transformation. To calculate the vehicle position I calculated the offset of the center between the two lane line bases with the center of the image in horizontal direction (code lines 41 and 42), and then multiplied with the pixel to meters scale in the horizontal direction.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The entire `pipeline` is defined in the class `FrameProcessor` in the file named `FrameProcessor.py` (code lines 90 through 100) that takes in a raw image captured from the camera and returns the annotated image. 
```python
    def pipeline(self, image):
        original = image.copy()
        image = self.corrector.correct(image)
        image = self.aerializer.aerialize(image)
        image = self.roi.crop(image) #region of onterest
        image = self.roi.create_binary(image)
        image = self.roi.decrop(image)
        image = self.annotation(image)
        image = self.annotate(image, original)
        image = self.write_curvature(image)
        return image

```

To be able to annotate the lane region with the polygon obtained above on the original image, it is essential to perform a reverse perspective transformation of the polygon so that they become aligned to the lane lines of the original image. I performed this operation using the `aerializer.deaerialize()` method mentioned earlier. This is done inside the `annotation()` method of `FrameProcessor` (code lines 35 thorugh 50). I also wrote the curvature of the road and position of the vehicle on the lane obtained above on the image using the `write_curvature()` method of `FrameProcessor` (code lines 57 through 88). To make the text more "Robotic" I used Consolas.ttf font face.

The result obtained is as follows:

|Final Result           |
|:-------------------------:|
|![alt text][image6.0]      |


---

### Pipeline (video)

The `FrameProcessor` class has two additional methods `process_bgr()` and `process_rgb()` to process the image. Depending on whether the original image is BGR or RGB format, the respective method needs to be invoked. These methods internally calls the `pipeline()` to process and return the annotated image.

The `Main.py` file contains a method named `process_video()` which takes the video file path and the path where the annotated video will be saved. This method internally calls the `process_rgb()` of `FrameProcessor` as the video frames passed are in RGB format. This method processes the entire video and saves the annotated video in the mentioned location.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/uniquetrij/CarND-P4-Advanced-Lane-Lines/blob/master/project_video_annotated.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Although this approach worked fine on the project video, it didn't perform too well on the challenge videos. This might work better if more filters on other color channels are used along with sobel operator. Although I implemented and tested the sobel operator, I didn't use it in the final pipeline as it didn't provide any improvement for annotating the project video.

2. Selection of the scaling value for pixel to meters conversion was mostly based on trial and error to obtain an approximate value of the curvature radius (which is given as 1km). 

3. A times, the lane annotation wobbles sharply between frames of the video. This may be smoothened out by taking the mean of past few frames.

4. VideoFileClip loads video frames in RGB while the vc2 expects them to be BGR. As a result, this conversion had to be taken care of.

5. It seemed to me that contracting the image during perspective transformation instead of expanding gave better results as there was almost no blurring of the lane lines which would otherwise have occured due to interpolation.


