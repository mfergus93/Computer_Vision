# Computer Vision HW4
import cv2
import numpy as np

def fillarea(ctr):
    maxx = np.max(ctr[:, 0]) + 1
    maxy = np.max(ctr[:, 1]) + 1
    contourImage = np.zeros( (maxy, maxx) )
    length = ctr.shape[0]
    for count in range(length):
        contourImage[ctr[count, 1], ctr[count, 0]] = 255
        cv2.line(contourImage, (ctr[count, 0], ctr[count, 1]), \
        (ctr[(count + 1) % length, 0], ctr[(count + 1) % length, 1]), \
        (255, 0, 255), 1)
    fillMask = cv2.copyMakeBorder(contourImage, 1, 1, 1, 1, \
    cv2.BORDER_CONSTANT, 0).astype(np.uint8)
    areaImage = np.zeros((maxy, maxx), np.uint8)
    startPoint = (int(maxy/2), int(maxx/2))
    cv2.floodFill(areaImage, fillMask, startPoint, 128)
    area = np.sum(areaImage)/128
    return area

def pavlidis(img):
    x,c=0,0
    y=img.shape[0]//2

    while c == 0:
        x+=1
        if img[y,x]!=0:
            c=1

    b1_value,b2_value,b3_value = 0,0,0
    b1_coord,b2_coord,b3_coord= 0,0,0
    directions = 'up', 'right', 'down', 'left'
    direction='right'
    input=(y,x)
    result=np.array([input])

    c=0
    start_pos=input
    while c<2:

        y,x=input
        if input==list(start_pos):
            c=c+1

        if direction =='up':
            b1_value, b1_coord=img[y-1,x-1], [y-1,x-1]
            b2_value, b2_coord=img[y-1,x], [y-1,x]
            b3_value, b3_coord=img[y-1,x+1], [y-1,x+1]
        elif direction =='right':
            b1_value, b1_coord=img[y-1,x+1], [y-1,x+1]
            b2_value, b2_coord=img[y,x+1], [y,x+1]
            b3_value, b3_coord=img[y+1,x+1], [y+1,x+1]
        elif direction =='down':
            b1_value, b1_coord=img[y+1,x+1], [y+1,x+1]
            b2_value, b2_coord=img[y+1,x], [y+1,x]
            b3_value, b3_coord=img[y+1,x-1], [y+1,x-1]
        elif direction =='left':
            b1_value, b1_coord=img[y+1,x-1], [y+1,x-1]
            b2_value, b2_coord=img[y,x-1], [y,x-1]
            b3_value, b3_coord=img[y-1,x-1], [y-1,x-1]

        block_values= b1_value,b2_value,b3_value
        block_coords=b1_coord,b2_coord,b3_coord

        if b1_value==255:
            bimg[y,x]=255
            direction=directions[(((directions.index(direction)-1))%4)]
        elif b1_value==0 and b2_value==0 and b3_value==0:
            direction=directions[(((directions.index(direction)+1))%4)]
        elif b2_value==255:
            bimg[y,x]=255
        elif b3_value==255:
            bimg[y,x]=255

        for i, value in enumerate(reversed(block_values)):
            if value==255:
                input = block_coords[2-i]

        result=np.append(result,[input],axis=0)

    result, ind=np.unique(result,axis=0, return_index=True)
    result=result[np.argsort(ind)]
    return (result)

def discrete_curve_evolution(x,y):
    dce_theta= np.arctan((y-np.roll(y,1))/(x-np.roll(x,1)))-np.arctan((np.roll(y,-1)-y)/(np.roll(x,-1)-x))
    dce_relevance_num=np.abs(dce_theta)*np.sqrt((x-np.roll(x,1))**2+(y-np.roll(y,1))**2)*np.sqrt((np.roll(x,-1)-x)**2+(np.roll(y,-1)-y)**2)
    dce_relevance_denom=np.sqrt((x-np.roll(x,1))**2+(y-np.roll(y,1))**2)+np.sqrt((np.roll(x,-1)-x)**2+(np.roll(y,-1)-y)**2)
    dce_relevance=dce_relevance_num/dce_relevance_denom

    dce_relevance_array=np.column_stack((dce_relevance,x,y))
    remove_index=dce_relevance_array[:,0].argmin()
    dce_relevance_array=np.delete(dce_relevance_array, remove_index, axis=0)

    x=dce_relevance_array[:,1]
    y=dce_relevance_array[:,2]
    dce_contour=dce_relevance_array[:,1:]
    return(x,y, dce_contour)

def gaussarea(x,y):
    gauss_area=0.5*np.abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))
    return(gauss_area)

files=['hand0','US']
for file in files:

    o_img=cv2.imread(r'C:\Users\Matt\OneDrive\Virginia Tech\CV\Contour\\'+file+'.png')
    img=np.mean(o_img.copy(),2)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=img.astype(np.uint8)
    threshold,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,None,value=(0,0,0))
    img_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Results\Binary_'+file+'.png',img)
    bimg=np.zeros(img.shape[:2])

    contour=pavlidis(img)

    xc=np.float64(contour[:,1])
    yc=np.float64(contour[:,0])

    print('Filename: '+file+'.png') #File Name
    print(contour.shape[0]) #Vertices
    print(str(gaussarea(xc,yc))) #Gauss Area
    print(str(fillarea(np.int32(np.concatenate((xc.reshape(-1,1),yc.reshape(-1,1)),axis=1))))) #Flood Area

    for i in range(1,9):

        x=xc.copy()
        y=yc.copy()

        for j in range(int(contour.shape[0]*(1-1/2**i))):

            x,y,dce_points=discrete_curve_evolution(x,y)

        print(str(dce_points.shape[0])) #Vertices
        print(str(gaussarea(x,y))) #Gauss Area
        print(str(fillarea(np.int32(np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1))))) #Flood Area

        dce_points=dce_points.astype(np.int32)
        blank_img_2=np.zeros(img.shape).astype(np.int32)
        dce_img=cv2.polylines(blank_img_2,[dce_points], True, (255,255,255),1)
        cv2.imwrite(r'C:\Users\Matt\Desktop\Results\\'+file+'DCE_'+str(i)+'.png',dce_img)


