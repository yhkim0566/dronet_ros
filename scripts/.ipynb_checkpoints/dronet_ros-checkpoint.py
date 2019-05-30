#! /usr/bin/python
#import tensorflow as tf
#import keras
#from keras.backend.tensorflow_backend import set_session

import numpy as np
import time
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS
class zedRosDronet:
    def __init__(self):
        
        self.img = CompressedImage()
        self.grayimg = None
        self.img_sub = rospy.Subscriber('/zed/zed_node/left/image_rect_color/compressed',
                                        CompressedImage, self.img_callback)

        

        
    def img_callback(self, data):
        self.img = data
        self.bridge = CvBridge()
        temp = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        self.grayimg = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow("Image", self.grayimg)
        cv2.waitKey(1)
        
        
def main():
    
    getimg = zedRosDronet()
    rospy.init_node('dronet_node', anonymous=True)
    
    rate = rospy.Rate(30)
    
    while True:
        rate.sleep()
        
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
        



    
#print(rgb_img)
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)

# Best
json_path = './model/model_struct.json'
weights_path = './model/best_weights.h5'

# For drone control
#json_path = './models2/model_struct.json'
#weights_path = './models2/model_weights_59.h5'

loaded_json = open(json_path, 'r')
loaded_model = loaded_json.read()

model = keras.models.model_from_json(loaded_model)
model.load_weights(weights_path)
"""
