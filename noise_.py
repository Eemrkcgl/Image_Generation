import random
import pybboxes as pbx

def Noise(img,bbox,perc): 
  
    # Getting the dimensions of the image 
        
    for box in bbox:
        _, x, y, w, h = map(float, box.strip().split(' '))
        pbx.convert_bbox((x,y,w,h), from_type="yolo", to_type="voc", image_size=(W, H))
      
        # Randomly pick some pixels in the 
        # image for coloring them white 
        # Pick a random number between 300 and 10000 
        number_of_pixels =  w*h*perc/100
        for i in range(int(number_of_pixels/2)): 
            
            # Pick a random y coordinate 
            y_coord=random.randint(x-w/2, x+w/2) 
            
            # Pick a random x coordinate 
            x_coord=random.randint(y-h/2, y+h/2) 
            
            # Color that pixel to white 
            img[y_coord][x_coord] = 255
            
        
        for i in range(int(number_of_pixels/2)): 
            
            # Pick a random y coordinate 
            y_coord=random.randint(x-w/2, x+w/2)
            
            # Pick a random x coordinate 
            x_coord=random.randint(y-h/2, y+h/2) 
            
            # Color that pixel to black 
            img[y_coord][x_coord] = 0
            
    return (img,bbox)