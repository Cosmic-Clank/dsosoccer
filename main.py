import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
from game import Game
import cv_viewer.tracking_viewer as cv_viewer


def main():
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
    # obj_param.enable_mask_output = False  # TEST THIS
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    # obj_param.prediction_timeout_s = 0.1  # TEST THIS
    # obj_param.image_sync = True # TEST THIS
    # obj_param.allow_reduced_precision_inference = True  # TEST THIS
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
        sl.OBJECT_CLASS.PERSON: 60, sl.OBJECT_CLASS.SPORT: 0}

    image_left_zed = sl.Mat()
    
    game = Game()
    # game = None

    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        err = zed.retrieve_objects(objects, obj_runtime_param)

        zed.retrieve_image(image_left_zed, sl.VIEW.LEFT)
        image_left_ocv_without_inference = image_left_zed.get_data()
        image_left_ocv_with_inference = image_left_ocv_without_inference.copy()

        cv_viewer.render_2D(image_left_ocv_with_inference, objects, obj_param.enable_tracking)
        
        if game:
            football_map = game.generate_football_map(image_left_ocv_without_inference, objects, obj_param.enable_tracking)
            scoreboard = game.generate_scoreboard()
            
            resized_football_map = cv2.resize(football_map, (360, 360), interpolation=cv2.INTER_AREA)
            resized_scoreboard = cv2.resize(scoreboard, (360, 360), interpolation=cv2.INTER_AREA)

            game_board = cv2.vconcat([resized_football_map, resized_scoreboard])
            resized_image_left_ocv_with_inference = cv2.resize(image_left_ocv_with_inference, (1280, 720), interpolation=cv2.INTER_AREA)

            cv2.imshow("Football ground", cv2.hconcat([game_board, cv2.cvtColor(resized_image_left_ocv_with_inference, cv2.COLOR_BGRA2BGR)]))
        else:
            resized_image_left_ocv_with_inference = cv2.resize(image_left_ocv_with_inference, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.imshow("Image", resized_image_left_ocv_with_inference)
        

        FPSCOUNT += 1
        FPSSUM += zed.get_current_fps()

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Average FPS: ", FPSSUM / FPSCOUNT)
    cv2.destroyAllWindows()
    image_left_zed.free(memory_type=sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    sys.exit()


if __name__ == "__main__":
    main()
