# Computer Vision HW3
import cv2
import numpy as np

files='AllmanBrothers','CalvinAndHobbes','Chartres','Elvis1956'
for file in files:
    print(file+'.png')    
    original_img = cv2.imread(r'C:\Users\Matt\OneDrive\Virginia Tech\CV\Corner Detection\\'+file+'.png')
    original_img=original_img.astype(np.float32)
    img=np.mean(original_img.copy(),2)
    img=cv2.GaussianBlur(img,(15,15),0)  
    
    # Shi Tomasi Library Function
    corners=cv2.goodFeaturesToTrack(img,100,0.01,10)
    corners=corners[:,0,:]
    print(corners[:3])
    corners=np.intp(corners)
    
    shit_corner_img=original_img.copy()
    for i in corners:
        x,y=i.ravel()
        cv2.circle(shit_corner_img,(x,y),5,(0,254,0))
    
    img_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Results\\'+file+'Shi_Tomasi_Corners.png', shit_corner_img)
    
    # Harris Function
    harris_corner_img=original_img.copy()
    img=np.mean(original_img,2)
    
    ix=cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    iy=cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize =5)
    ixx=ix**2
    iyy=iy**2
    ixy=ix*iy
    ixx=cv2.GaussianBlur(ixx,(3,3),1)
    iyy=cv2.GaussianBlur(iyy,(3,3),1)
    ixy=cv2.GaussianBlur(ixy,(3,3),1)
    corner_map=(ixx*iyy-ixy**2)-0.05*(ixx+iyy)**2
    
    dX, dY = 10,10
    M, N = img.shape
    for x in range(0,M-dX+1):
        for y in range(0,N-dY+1):
            window = corner_map[x:x+dX, y:y+dY]
            if np.sum(window)==0:
                localMax=0
            else:
                localMax = np.amax(window)
            maxCoord=np.unravel_index(np.argmax(window), window.shape) + np.array((x,y))
            #suppress everything
            corner_map[x:x+dX, y:y+dY]=0
            #reset only the max
            corner_map[tuple(maxCoord)] = localMax
    
    row,column=img.shape[:2]
    max_corner_map=np.empty((3,row*column))
    counter=0
    for r in range(row):
        for c in range(column):
            max_corner_map[:,counter] = [corner_map[r,c],r,c]
            counter+=1
    
    max_corner_map = np.transpose(max_corner_map)
    max_corner_map = max_corner_map[~np.all(max_corner_map == 0, axis=1)]
    max_corner_map= max_corner_map[max_corner_map[:, 0].argsort()]
    max_corner_map=max_corner_map[-100:]
    print(np.fliplr(max_corner_map[-3:,1:]))
    
    for r in range(100):
            y=int(max_corner_map[r,1])
            x=int(max_corner_map[r,2])
            cv2.circle(harris_corner_img, (x,y),5,(0,0,254))
    img_out_2=cv2.imwrite(r'C:\Users\Matt\Desktop\Results\\'+file+'_Harris_Corners.png',harris_corner_img)
    
    output=np.concatenate((shit_corner_img,harris_corner_img),axis=1)
    img_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Results\\'+file+'_Concatenate.png',output)
    
    