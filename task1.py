"""
Created on Sat Feb  1 15:02:37 2020

@author: shawk
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Canny(image):
        gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur=cv2.GaussianBlur(gray, (5,5),0)
        canny=cv2.Canny(blur,50,150)
        return canny
def region_of_interest(image):
    height=image.shape[0]-50
    ploygons=np.array([[
            (200,height),(900,height),(400,550),(450,550)
            ]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,ploygons,255)
    masked_image=cv2.bitwise_and(image, mask)
    return masked_image

def lines_disply(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 =line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2),(0,0,255),10)
    return line_image
def make_coordinators(image,line_parameters):
    global negt
    global negt_1
    try:
        slop, intercept = line_parameters
    except TypeError:
        slop, intercept = 1000,1000
    if slop < 0:
        negt = slop
        negt_1=intercept
    elif slop == 1000:
        slop = negt
        intercept=negt_1
        
    y1 = image.shape[0]
    y2 = int(y1*0.85)
    x1 = int((y1-intercept)/slop)
    x2 = int((y2-intercept)/slop)
    print(slop)

    return np.array([x1,y1,x2,y2])

#funcation calculates the average of slop and intercept of interested lines with inputs image to use its diemention and vairable fo lines that have Xs and Ys of interees lines
#first step calucakte the slop and intercept of each line seperet ecah of them as left and right line to indecate the road, calculate the avaerage of them and retrun it in X and Y form 
#theoutputs is a array consists the average slope and intercept
def average_slope_intercept(image,Lines):
    left_fit=[]
    right_fit=[]
    for line in Lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slop=parameters[0]
        intercept=parameters[1]
        if slop < 0:
            left_fit.append((slop,intercept))
        else:
            right_fit.append((slop,intercept))
    average_left_fit=np.average(left_fit, axis=0)
    average_right_fit=np.average(right_fit, axis=0)
    left_line=make_coordinators(image,average_left_fit)
    right_line=make_coordinators(image,average_right_fit)
    return np.array([left_line, right_line])

#image=cv2.imread('test.jpg')
#copy=np.copy(image)
##canny=Canny(copy)
##masked=region_of_interest(canny)
##Lines=cv2.HoughLinesP(masked , 2, np.pi/180, 80, np.array([]), minLineLength=60, maxLineGap=20 )
##average_lines=average_slope_intercept(copy,Lines)
##Lined_image=lines_disply(image, average_lines)
##Final=cv2.addWeighted(copy,0.8,Lined_image,1,1)
##cv2.imshow('Result', Final)
##cv2.waitKey(0)

cap=cv2.VideoCapture("Lane Detection Test Video 01.mp4")
backup_Lines=0

while(cap.isOpened()):
    _, frame = cap.read()
    if frame is not None:
        canny=Canny(frame)
        Cropped_image=region_of_interest(canny)
        Lines=cv2.HoughLinesP(Cropped_image ,6, np.pi/360, 80, np.array([]), minLineLength=60, maxLineGap=20 )
        if Lines is None:
            average_lines=average_slope_intercept(frame,backup_Lines)
            lined_image=lines_disply(frame,average_lines)
            Final=cv2.addWeighted(frame,1,lined_image,1,1)
            cv2.imshow('Result', Cropped_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#            plt.imshow( lined_image)
#            plt.show()
        else:
            average_lines=average_slope_intercept(frame,Lines)
            lined_image=lines_disply(frame,average_lines)
            Final=cv2.addWeighted(frame,1,lined_image,1,1)
            cv2.imshow('Result', Final)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            backup_Lines=Lines
#            plt.imshow( Cropped_image)
#            
#            plt.show()
    else:
        cap.release()
        cv2.destroyAllWindows()

#
#plt.imshow( canny)
#plt.show()
