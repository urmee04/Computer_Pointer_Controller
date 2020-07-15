'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import cv2
import math
import os

class GazeEstimate:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        
        self.core = None
        self.network = None
        self.exec_network = None
        self.device = device
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

        self.core = IECore()
        self.network = IENetwork(model=str(model_name),
                                              weights=str(os.path.splitext(model_name)[0] + ".bin"))

        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

       

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)

        return new_mouse_coord, gaze_vector


    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Please check extention for these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported !!")

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        le_image_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        re_image_resized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
        le_img_processed = np.transpose(np.expand_dims(le_image_resized,axis=0), (0,3,1,2))
        re_img_processed = np.transpose(np.expand_dims(re_image_resized,axis=0), (0,3,1,2))
        return le_img_processed, re_img_processed

    def preprocess_output(self, outputs,hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        rollValue = hpa[2]
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (newx,newy), gaze_vector