
# Computer Pointer Controller
The computer pointer controller project is designed to demonstrate the ability to run multiple models in the OpenVINO toolkit on the same machine and to coordinate the flow of data between those models. The ultimate goal of this project is to control the mouse pointer by using a gaze detection model. 

## Project SetUp and Installation
The Inference Engine API from Intel's OpenVino ToolKit is required to build this project. 
 
- part1: Set up local development environment
1. Download and install the OpenVINO toolkit
2. Create a virtual environment for the project
3. Download and unzip the starter files
4. Setup webcam or locate the video file
5. Initialize the openVINO environment:-
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

- Part2: Download the models

The gaze estimation model requires three inputs:
1. The head pose
2. The left eye image
3. The right eye image.

Required models can be downloaded using the model downloader. To navigate to the directory containing the Model Downloader:
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

The following models are needed to run the project:

1. Face Detection Model
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001

2. Facial Landmarks Detection Model
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009

3. Head Pose Estimation Model
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001

4. Gaze Estimation Model
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002


- Part3: Build the Inference Pipeline
Using the Inference Engine API, it is required to build a pipeline that runs inference on the models using the inputs.

1. Loading the model
2. Preprocessing the inputs
3. Inference
4. Preprocessing the outputs

## Demo
Open a new terminal and run the main.py file by the following command:-
python main.py -fd /models/face-detection-adas-binary-0001.xml -fl /models/landmarks-regression-retail-0009.xml -hp /models/head-pose-estimation-adas-0001.xml -ge /models/gaze-estimation-adas-0002.xml -i resources/demo.mp4 

## Documentation

Command line arguments to run the project:-

-fd (required): path to Face Detection model's xml file
-fl (required): Path to Facial Landmark Detection mode's xml file.
-hp (required): path to Head Pose Estimation model's xml file
-ge (required): path to Gaze Estimation model's xml file
-i (required): specify the path of input video file or webcam
-flags (optional): specify the flags from fd, fl, hp, ge to visualize the output of corresponding models of each frame (write flags with space separation; -flags fd fld hp).
-l (optional): specify the absolute path of cpu extension if some layers of models are not supported on the device.
-prob (optional): specify the probability threshold for face detection model.
-d(optional): set the target device- CPU, GPU, FPGA or MYRIAD.


## Directory structure of the project:-

- resources/demo.mp4
- face_detection.py
- facial_landmarks_detection.py
- gaze_estimation.py
- head_pose_estimation.py
- input_feeder.py
- main.py
- mouse_controller.py
- requirements.text

###### face_detection.py
 Take a video frame as input,  perform inference on it and detect the face.
###### facial_landmarks_detection.py
 Take the detected face as input, preprocess it, perform inference on it and detect the eye landmarks.
###### gaze_estimation.py
 Take left eye, right eye, head pose angles as inputs, preprocess it, perform inference and predict the gaze vector.
###### head_pose_estimation.py
 Take the detected face as input, preprocess it, perform inference on it and detect the head position by predicting yaw, roll, pitch angles, postprocess the outputs.
###### input_feeder.py
 Contains InputFeeder class which initialize Video Capture and return the frames.
###### main.py
 Requires to run the main.py file for running the app.
###### mouse_controller.py
 Contains MouseController class which takes x, y coordinates value, speed, precisions and according to these values it moves the mouse pointer by using pyautogui library.
