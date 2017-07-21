# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import cv2

WIDTH = 32
HEIGHT = 32
DEPTH = 3

class fileReader():
    def __init__(self, filename):
        if not os.path.exists(filename):
            print(filename + ' is not exist')
            return
           
        self.bytestream = open(filename, mode="rb")
        self.dict = pickle.load(self.bytestream, encoding="bytes")
        self.numbers = len(self.dict[b'labels'])

    def read_file(self,flag):
        label_list = []
        image_list = []
        for index in range(self.numbers):
            label = self.dict[b'labels'][index]
            if flag==False and label==0 or flag==True and label==8:
                if label==8:
                    label = 1
                image = np.transpose(np.reshape(self.dict[b'data'][index*3:index*3+3], [DEPTH, WIDTH, HEIGHT]), [1, 2, 0])
                
                new_img = np.asarray(image.flatten())/255
                
                label_list.append(label)
                image_list.append(new_img)
            
        return (label_list, image_list)
        
    def read_image(self,index):
        label = self.dict[b'labels'][index]
        image = np.transpose(np.reshape(self.dict[b'data'][index*3:index*3+3], [3, 32, 32]), [1, 2, 0])
        
if __name__ == '__main__':
    print('===== Start Test of fileReader =====')
    reader = fileReader('prototype_1/test_batch')
    l,i = reader.read_file(False)
    for p in range(5):
        cv2.imwrite(str(p)+'.jpg',i[p])
        cv2.imshow('test',i[p])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('===== End Test of fileReader =====')
