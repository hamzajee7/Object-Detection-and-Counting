import cv2
import numpy as np

 # web camera
cap = cv2.VideoCapture('video.mp4')
# minium width and heigh of rectangle ...
min_width_rectangle = 80
min_height_rectangle = 80
# for line ...
count_line_position = 550
#initialize substrator
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
#Function for the counting ...
def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy
detect = []
offset = 6 #allowable error between pexel
counter = 0


while True:
    ret,frame1= cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applaying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    CounterShape,h = cv2.findContours(dilatada , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     # For to draw line we have ...
    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,17,0),3)

    # to make rectangle on the vehicle...
    for (i,c) in enumerate(CounterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rectangle) and (h>= min_height_rectangle)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        #for to give name to the vehicle ...
        # cv2.putText(frame1,"VEHICLE"+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,2,(255,244,0),2)


        #For counting ....
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                detect.remove((x,y))
                print("Vehicle Counter:"+str(counter))

    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)


    # cv2.imshow('Detector',dilatada)
    cv2.imshow('Video show',frame1)

    k = cv2.waitKey(30) & 0xff
    if k == 13:
        break

cv2.destroyAllwindows()
cap.release()
