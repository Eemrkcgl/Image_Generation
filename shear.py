import random
import cv2
import numpy as np
    
def HorizontalFlip(img,bboxes):
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    
    img =  img[:,::-1,:]
    bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
    
    box_w = abs(bboxes[:,0] - bboxes[:,2])
        
    bboxes[:,0] -= box_w
    bboxes[:,2] += box_w
    
    return img, bboxes

def Shearing(img, bboxes,shear_factor,type):

    shear_factor = random.uniform(shear_factor)

    w,h = img.shape[1], img.shape[0]

    if shear_factor < 0:
        img, bboxes = HorizontalFlip(img, bboxes)

    M = np.array([[1, abs(shear_factor), 0],[0,1,0]])

    nW =  img.shape[1] + abs(shear_factor*img.shape[0])

    bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 

    img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

    if shear_factor < 0:
        img, bboxes = HorizontalFlip(img, bboxes)

    img = cv2.resize(img, (w,h))

    scale_factor_x = nW / w

    bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 
    
    return img, bboxes