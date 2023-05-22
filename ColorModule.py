"""
Color Module
Finds color in an image based on hsv values
Can run as stand alone to find relevant hsv values

"""
import pyzed.sl as sl
import cv2
import numpy as np
import logging
import sys

class ColorFinder:
    def __init__(self, trackBar=False):
        self.trackBar = trackBar
        if self.trackBar:
            self.initTrackbars()

    def empty(self, a):
        pass

    def initTrackbars(self):
        """
        To intialize Trackbars . Need to run only once
        """
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, self.empty)
        cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, self.empty)
        cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, self.empty)
        cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, self.empty)
        cv2.createTrackbar("Val Min", "TrackBars", 0, 255, self.empty)
        cv2.createTrackbar("Val Max", "TrackBars", 255, 255, self.empty)

    def getTrackbarValues(self):
        """
        Gets the trackbar values in runtime
        :return: hsv values from the trackbar window
        """
        hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
        smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
        vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
        hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
        smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
        vmax = cv2.getTrackbarPos("Val Max", "TrackBars")

        hsvVals = {"hmin": hmin, "smin": smin, "vmin": vmin,
                   "hmax": hmax, "smax": smax, "vmax": vmax}
        # print(hsvVals)
        return hsvVals

    def update(self, img, myColor=None):
        """
        :param img: Image in which color needs to be found
        :param hsvVals: List of lower and upper hsv range
        :return: (mask) bw image with white regions where color is detected
                 (imgColor) colored image only showing regions detected
        """
        imgColor = [],
        mask = []

        if self.trackBar:
            myColor = self.getTrackbarValues()

        if isinstance(myColor, str):
            myColor = self.getColorHSV(myColor)

        if myColor is not None:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([myColor['hmin'], myColor['smin'], myColor['vmin']])
            upper = np.array([myColor['hmax'], myColor['smax'], myColor['vmax']])
            mask = cv2.inRange(imgHSV, lower, upper)
            imgColor = cv2.bitwise_and(img, img, mask=mask)
        return imgColor, mask

    def getColorHSV(self, myColor):

        if myColor == 'red':
            output = {'hmin': 146, 'smin': 141, 'vmin': 77, 'hmax': 179, 'smax': 255, 'vmax': 255}
        elif myColor == 'green':
            output = {'hmin': 44, 'smin': 79, 'vmin': 111, 'hmax': 79, 'smax': 255, 'vmax': 255}
        elif myColor == 'blue':
            output = {'hmin': 103, 'smin': 68, 'vmin': 130, 'hmax': 128, 'smax': 255, 'vmax': 255}
        else:
            output = None
            logging.warning("Color Not Defined")
            logging.warning("Available colors: red, green, blue ")

        return output


def main():
    myColorFinder = ColorFinder(True)
    FPSCOUNT = 0
    FPSSUM = 0
    
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = 60  # Set fps at 60
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.sdk_verbose = True

    # Read from an svo file if file path provided in command line
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit(1)

    runtime_params = sl.RuntimeParameters()

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_mask_output = False  # TEST THIS
    obj_param.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    # obj_param.prediction_timeout_s = 0.1  # TEST THIS
    # obj_param.image_sync = True # TEST THIS
    obj_param.allow_reduced_precision_inference = True  # TEST THIS
    # obj_param.filtering_mode = sl.OBJECT_FILTERING_MODE.NMS3D_PER_CLASS
    obj_param.enable_tracking = True
    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40
    obj_runtime_param.object_class_filter = [
        sl.OBJECT_CLASS.PERSON, sl.OBJECT_CLASS.SPORT]
    obj_runtime_param.object_class_detection_confidence_threshold = {
        sl.OBJECT_CLASS.PERSON: 60, sl.OBJECT_CLASS.SPORT: 20}

    image_left_zed = sl.Mat()
    

    # Custom Orange Color
    hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        err = zed.retrieve_objects(objects, obj_runtime_param)

        zed.retrieve_image(image_left_zed, sl.VIEW.LEFT)
        img = image_left_zed.get_data()
        
        imgRed, _ = myColorFinder.update(img, "red")
        imgGreen, _ = myColorFinder.update(img, "green")
        imgBlue, _ = myColorFinder.update(img, "blue")
        imgOrange, _ = myColorFinder.update(img, hsvVals)

        cv2.imshow("Red", imgOrange)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(myColorFinder.getTrackbarValues())
    cv2.destroyAllWindows()
    image_left_zed.free(memory_type=sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    sys.exit()
if __name__ == "__main__":
    main()
