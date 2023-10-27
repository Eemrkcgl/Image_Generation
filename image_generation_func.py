import cv2
import random
import numpy as np
import os
import pybboxes as pbx

def label_reader(dir_of_file):
    bboxes=[]
    with open(dir_of_file) as f:
        for each in f:
            bboxes.append(each.strip().split(' '))
    return bboxes

def GammaSaturation(img,bboxes,saturation):
    saturation=(-saturation,saturation)
    saturation=random.randint(saturation)
    img = img.astype(int)
    kernel = np.array([saturation]).astype(int)
    img += np.reshape(kernel, (1,1,1))
    img = np.clip(img, 0, 255)
    img[:,:,0] = np.clip(img[:,:,0],0, 179)
    img = img.astype(np.uint8)
    return img, bboxes

def noise_bboxes(img,bbox,perc,W,H): 
    for box in bbox:
        _, x, y, w, h = map(float, box.strip().split(' '))
        pbx.convert_bbox((x,y,w,h), from_type="yolo", to_type="voc", image_size=(W, H))
        number_of_pixels =  w*h*perc/100
        for i in range(int(number_of_pixels/2)): 
            y_coord=random.randint(x-w/2, x+w/2) 
            x_coord=random.randint(y-h/2, y+h/2) 
            img[y_coord][x_coord] = 255
        for i in range(int(number_of_pixels/2)): 
            y_coord=random.randint(x-w/2, x+w/2)
            x_coord=random.randint(y-h/2, y+h/2) 
            img[y_coord][x_coord] = 0
    return (img,bbox)

def blur_bboxes(img,bbox,W,H):
    img_arr=img
    for box in bbox:
        _, x, y, w, h = map(float, box.strip().split(' '))
        pbx.convert_bbox((x,y,w,h), from_type="yolo", to_type="voc", image_size=(W, H))
        roi = img[x-w/2:x+w/2, y-h/2:y+h/2]
        blur_image = cv2.GaussianBlur(roi,(5,5),0)
        img_arr[x-w/2:x+w/2, y-h/2:y+h/2] = blur_image
    return (img_arr, bbox) 

def HorizontalFlip(img,bboxes):
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    img =  img[:,::-1,:]
    bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
    box_w = abs(bboxes[:,0] - bboxes[:,2])
    bboxes[:,0] -= box_w
    bboxes[:,2] += box_w
    return (img, bboxes)

def Shearing(img, bboxes,shear_factor):
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
    return (img, bboxes)

def random_image_generator(dir_of_the_images,dir_of_the_labels,wanted_number_of_new_generated_images,image_output_dir,label_output_dir):
    list_of_images=os.listdir(dir_of_the_images)
    number_of_images=len(list_of_images)
    generated_indexes=[]
    agumentation_numbers={"Shear":0,"Blur":0,"Noise":0,"GammaSaturation":0}
    agumentations={0:"Shear",1:"Blur",2:"Noise",3:"GammaSaturation"}
    for i in range(wanted_number_of_new_generated_images):
        index_of_random_image=random.randint(0,number_of_images-1)
        if index_of_random_image not in generated_indexes:
            image=cv2.imread(dir_of_the_images+list_of_images[index_of_random_image])
            H, W = image.shape[:2]
            coordinates=label_reader(dir_of_the_labels+list_of_images[index_of_random_image][:-4]+".txt")
            agumentation_type=random.randint(0,3)
            
            if agumentation_type==0:#Shear
                counter=0
                for coordinate in coordinates:
                    counter+=1
                    type_of_label:str=coordinate[0]
                    coordinate:list=coordinate[1:]
                    result=Shearing(image,coordinate,15)
                    cv2.imwrite(image_output_dir+list_of_images[index_of_random_image][:-4]+'_share{}.jpeg'.format(counter),result(0))
                    with open(label_output_dir+list_of_images[index_of_random_image][:-4]+'_share{}.txt'.format(counter),'w') as file:
                        file.write(coordinate[0]+' '+str(result[1][0])+' '+str(result[1][1])+' '+str(result[1][2])+' '+str(result[1][3])+' '+str(result[1][4]))
                        
            elif agumentation_type==1:#Blur
                result=blur_bboxes(image,coordinate,H,W)
                cv2.imwrite(image_output_dir+list_of_images[index_of_random_image][:-4]+'_blur.jpeg',result(0))
                with open(label_output_dir+list_of_images[index_of_random_image][:-4]+'_blur.txt','w') as file:
                    for each in result[1]:
                        for every in each:
                            file.write(every)
                        file.write('\n')
                    
            elif agumentation_type==2:#Noise
                result=noise_bboxes(image,coordinates,5,W,H)
                cv2.imwrite(image_output_dir+list_of_images[index_of_random_image][:-4]+'_noise.jpeg',result(0))
                with open(label_output_dir+list_of_images[index_of_random_image][:-4]+'_noise.txt','w') as file:
                    for each in result[1]:
                        for every in each:
                            file.write(every)
                        file.write('\n')
                
            elif agumentation_type==3:#GammaSaturation
                result=GammaSaturation(image,coordinates,30)
                cv2.imwrite(image_output_dir+list_of_images[index_of_random_image][:-4]+'_saturation.jpeg',result(0))
                with open(label_output_dir+list_of_images[index_of_random_image][:-4]+'_saturation.txt','w') as file:
                    for each in result[1]:
                        for every in each:
                            file.write(every)
                        file.write('\n')
            
            agumentation_numbers[agumentations[agumentation_type]]+=1
            generated_indexes.append(index_of_random_image)
        
        else:
            continue
if __name__=="__name__":
    dir_of_the_images=input('Enter the directory of images: ')
    wanted_number_of_new_generated_images=int(input('Enter the number of wanted new generated image: '))
    dir_of_the_labels=input('Enter the directory of labels: ')
    image_output_dir=input('Enter the directory of images: ')
    label_output_dir=input('Enter the directory of labels: ')
    random_image_generator(dir_of_the_images,dir_of_the_labels,wanted_number_of_new_generated_images,image_output_dir,label_output_dir)