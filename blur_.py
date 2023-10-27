import cv2
import pybboxes as pbx
def blur_bboxes(img,bbox,W,H):
    img_arr=img
    for box in bbox:
        _, x, y, w, h = map(float, box.strip().split(' '))
        pbx.convert_bbox((x,y,w,h), from_type="yolo", to_type="voc", image_size=(W, H))
        roi = img[x-w/2:x+w/2, y-h/2:y+h/2]
        blur_image = cv2.GaussianBlur(roi,(5,5),0)
        
        img_arr[x-w/2:x+w/2, y-h/2:y+h/2] = blur_image
        
    return (img_arr, bbox) 

            
    