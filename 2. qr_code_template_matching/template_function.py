import cv2
import numpy as np

w=99
x=10
p=(w-7*x)//2 #1/2 padding total, just one side of padding

template=np.zeros((x*7+2*p,x*7+2*p,3),np.uint8)

template=cv2.rectangle(template,(0,0),(w,w),(255,255,255),-1)
template=cv2.rectangle(template,(p,p),(w-p-1,w-p-1),(0,0,0),-1)
template=cv2.rectangle(template,(p+x,p+x),(w-p-1-x,w-p-1-x),(255,255,255),-1)
template=cv2.rectangle(template,(p+x+x,p+x+x),(w-p-1-x-x,w-p-1-x-x),(0,0,0),-1)

template_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\CV\template_E.png', template)


w=128
x=16
p=(w-7*x)//2

template=np.zeros((x*7+2*p,x*7+2*p,3),np.uint8)

template=cv2.rectangle(template,(0,0),(w,w),(255,255,255),-1)
template=cv2.rectangle(template,(p,p),(w-p-1,w-p-1),(0,0,0),-1)
template=cv2.rectangle(template,(p+x,p+x),(w-p-1-x,w-p-1-x),(255,255,255),-1)
template=cv2.rectangle(template,(p+x+x,p+x+x),(w-p-1-x-x,w-p-1-x-x),(0,0,0),-1)

template_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\CV\template_AD.png', template)


