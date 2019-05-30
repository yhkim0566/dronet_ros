#! /usr/bin/python
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from threading import Thread

import numpy as np
import time
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS
class zedRosDronet:
    def __init__(self, json_path, weights_path):
        
        self.img = CompressedImage()
        self.grayimg = None
        self.gray_resize = None
        self.img_size = (200, 200)
        self.img_sub = rospy.Subscriber('/zed/zed_node/left/image_rect_color/compressed',
                                        CompressedImage, self.img_callback)
        
        # DroNet
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.log_device_placement = True
        self.sess = tf.Session(config=self.config)
        self.setsess = set_session(self.sess)
        
        self.loaded_json = open(json_path, 'r')
        self.loaded_model = self.loaded_json.read()
        
        self.model = keras.models.model_from_json(self.loaded_model)
        self.model.load_weights(weights_path)
        
        """
        self.dronet_thread = Thread(target=self.prediction, args())
        self.dronet_thread.daemon = True
        self.dronet_thread.start()
        self.thead_flag = True
        self.isGet = False
        self.pred = None
        """
        
        
    def prediction(self):
        if self.isGet:
            self.pred = self.model.predict(self.gray_resize)
            print("Steering: ", self.pred[0][0,0], "Collision: ", self.pred[1][0,0])    

   
    def img_callback(self, data):
        self.img = data
        self.bridge = CvBridge()
        temp = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        self.grayimg = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        temp2 = cv2.resize(self.grayimg, self.img_size)
        self.gray_resize = temp2.reshape((1,200,200,1))
        
        self.isGet = True
        #cv2.imshow("Image", self.grayimg)
        #cv2.waitKey(1)
        
        
        
def main():
    json_path = './model/model_struct.json'
    weights_path = './model/best_weights.h5'
    dronet = zedRosDronet(json_path=json_path, weights_path=weights_path)
    rospy.init_node('dronet_node', anonymous=True)
    
    loop = rospy.Rate(30)
    while not rospy.is_shutdown():
        dronet.prediction()
        loop.sleep()
    
        
    #cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
        
