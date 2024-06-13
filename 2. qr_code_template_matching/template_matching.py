# ECE Homework 2 Finalized Version
import cv2
import numpy as np
from sklearn.cluster import KMeans

qrs=['A','B','C','D','E']
for qr in qrs:

    template_AD=cv2.imread(r'C:\Users\Matt\Desktop\Virginia Tech\CV\template_AD.png',0)
    template_E=cv2.imread(r'C:\Users\Matt\Desktop\Virginia Tech\CV\template_E.png',0)
    img = cv2.imread(r'C:\Users\Matt\Desktop\Virginia Tech\CV\QR_'+qr+'.png', 0)
    img_origin = cv2.imread(r'C:\Users\Matt\Desktop\Virginia Tech\CV\QR_'+qr+'.png')
    filename= r'QR_'+qr+'.png'
    img=img.astype(np.float32)
    template_AD=template_AD.astype(np.float32)
    template_E=template_E.astype(np.float32)
    
    # Blur masked image and template
    img=img.astype(np.float32)
    img=cv2.GaussianBlur(img,(15,15),0)
    template_AD=cv2.GaussianBlur(template_AD,(3,3),0)
    template_E=cv2.GaussianBlur(template_E,(3,3),0)
    
    res_AD=cv2.matchTemplate(img, template_AD, cv2.TM_CCORR_NORMED)
    res_E=cv2.matchTemplate(img, template_E, cv2.TM_CCORR_NORMED)
    min_val_AD, max_val_AD, min_loc_AD, max_loc_AD = cv2.minMaxLoc(res_AD)
    min_val_E, max_val_E, min_loc_E, max_loc_E = cv2.minMaxLoc(res_E)
    
    if max_val_AD>max_val_E:
        res=res_AD
    if max_val_AD<max_val_E:
        res=res_E
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    res=255*(res-min_val)/(max_val-min_val)
    
    h,w=img.shape[:2]
    cx,cy=w//2,h//2
    
    qr_max_arr=np.empty((3,(h-127)*(w-127)))
    counter=0
    for r in range(h-127):
        for c in range(w-127):
            if res[r,c]>200:
                qr_max_arr[:,counter] = [res[r,c], r, c]
                counter+=1
    
    qr_max_arr = np.transpose(qr_max_arr)
    qr_max_arr=qr_max_arr[~np.all(qr_max_arr == 0, axis=1)]
    qr_max_arr = qr_max_arr[qr_max_arr[:, 0].argsort()]
    qr_max_arr=qr_max_arr[-300:]
    
    # Kmeans clustering analysis of maxima
    kmean=KMeans(n_clusters=3)
    kfit=kmean.fit(X=qr_max_arr[:,1:])
    kpredict=kmean.predict(qr_max_arr[:,1:])
    
    centroids=(kmean.cluster_centers_)
    centroids=centroids.astype(int)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    cv2.rectangle(img,(centroids[0,1],centroids[0,0],),(centroids[0,1]+128,centroids[0,0]+128),(0,0,255),2)
    cv2.rectangle(img,(centroids[1,1],centroids[1,0]),(centroids[1,1]+128,centroids[1,0]+128),(0,0,255),2)
    cv2.rectangle(img,(centroids[2,1],centroids[2,0]),(centroids[2,1]+128,centroids[2,0]+128),(0,0,255),2)
    
    centroids=kmean.cluster_centers_
    xymean=centroids.mean(axis=0)
    xmean=xymean[1]
    ymean=xymean[0]
    
    for x in range(3):
        if centroids[x,1]<xmean and centroids[x,0]<ymean:
            a_x=centroids[x,1]+64
            a_y=centroids[x,0]+64
        elif centroids[x,1]>xmean and centroids[x,0]<ymean:
            b_x=centroids[x,1]+64
            b_y=centroids[x,0]+64
        elif centroids[x,1]<xmean and centroids[x,0]>ymean:
            c_x=centroids[x,1]+64
            c_y=centroids[x,0]+64
            
    A=50,50
    B=50,250
    C=250,50
    destinations=np.array([A,C,B]).astype(np.float32)
    destinations=np.ndarray.copy(destinations, order='C')
    source=np.array([[a_x,a_y],[b_x,b_y],[c_x,c_y]])
    source=np.ndarray.copy(source, order='C')
    source=source.astype(np.float32)
    warp_mat=cv2.getAffineTransform(source, destinations)
    warped=cv2.warpAffine(img_origin,warp_mat,(300,300))
    
    print(filename)
    print(centroids)
    img_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\CV\QR\Mask\qr_'+qr+'_res.png', res)
    mimg_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\CV\QR\Mask\qr_'+qr+'rect.png', img)
    warp_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Virginia Tech\CV\QR\Mask\warp'+qr+'.png', warped)
    
        
            
        