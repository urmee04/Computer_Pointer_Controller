import cv2
import logging as log
import os
import sys
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from argparse import ArgumentParser
from face_detection import FaceDetect
from facial_landmarks_detection import FacialDetect
from gaze_estimation import GazeEstimate
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder

def build_argparser():
   
    parser = ArgumentParser()
    parser.add_argument("-fd", "--facedetection", required=True, type=str,
                        help="The location of the XML file of Face Detection model.")
    parser.add_argument("-fl", "--facialdetection", required=True, type=str,
                        help="The location of the XML file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="The location of the XML file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="The location of the XML file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="The location of the input file")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fl, hp, ge like --flags fd hp fl (Seperate each flag by space)"
                             "fd for Face Detection, fl for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD"
                             "specified (CPU by default)")
    
    return parser


def main():

    # Command line args
    args = build_argparser().parse_args()
    previewFlags = args.previewFlags
    
    logger = log.getLogger()
    inputFilePath = args.input
    inputFeeder = None
    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to locate specified video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)
    
    modelPathDict = {'FaceDetect':args.facedetection, 'FacialDetect':args.facialdetection, 
    'GazeEstimate':args.gazeestimation, 'HeadPoseEstimation':args.headpose}
    
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified "+fileNameKey+" xml file")
            exit(1)
            
    fdm = FaceDetect(modelPathDict['FaceDetect'], args.device, args.cpu_extension)
    fldm = FacialDetect(modelPathDict['FacialDetect'], args.device, args.cpu_extension)
    gem = GazeEstimate(modelPathDict['GazeEstimate'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimation(modelPathDict['HeadPoseEstimation'], args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')
    
    inputFeeder.load_data()
    fdm.load_model()
    fldm.load_model()
    gem.load_model()
    hpem.load_model()
    
    frame_count = 0
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        if type(croppedFace)==int:
            logger.error("Face not detected.")
            if key==27:
                break
            continue
        
        hp_out = hpem.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)
        
        if (not len(previewFlags)==0):
            preview_frame = frame.copy()
            if 'fd' in previewFlags:
                preview_frame = croppedFace
            if 'fl' in previewFlags:
                cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
            if 'hp' in previewFlags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | Pitch:{:.2f} | Roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (239, 174, 0), 2)
            if 'ge' in previewFlags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re

                

        cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
       
        
        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
            break
    
    logger.error("VideoStreaming Ended")
    cv2.destroyAllWindows()
    inputFeeder.close()
    
if __name__ == '__main__':
    main()     
    
