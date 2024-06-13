#Computer Vision Homework 5 Matt Ferguson
import cv2
import numpy as np

files=['rio', 'blacksburg', 'diamondhead']
for file in files:

    # Loading
    img_1=cv2.imread(r'C:\Users\Matt\OneDrive\Virginia Tech\CV\Stitching\\'+file+'-00.png',0)
    img_2=cv2.imread(r'C:\Users\Matt\OneDrive\Virginia Tech\CV\Stitching\\'+file+'-01.png',0)
    img_3=cv2.imread(r'C:\Users\Matt\OneDrive\Virginia Tech\CV\Stitching\\'+file+'-02.png',0)

    imgs=[img_1,img_2,img_3]
    img_left=img_1

    for i in range(2):

        img_left_original_width=img_left.shape[1]
        img_right=imgs[i+1]

        # Feature Mapping
        sift_object=cv2.SIFT_create(500)
        kp_1, desc_1=sift_object.detectAndCompute(img_left, None)
        kp_2, desc_2=sift_object.detectAndCompute(img_right, None)
        bf=cv2.BFMatcher()
        bf_matches=bf.knnMatch(desc_1,desc_2, k=2)
        good = []
        for m,n in bf_matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        # Homography
        dst_pts=np.float32([ kp_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kp_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
        print(M)
        img_right=cv2.warpPerspective(img_right,M,((img_left.shape[1]+img_right.shape[1]),int(img_left.shape[0])))

        black_img=np.zeros(img_right.shape)
        black_img[0:img_left.shape[0],0:img_left.shape[1]]=img_left
        img_left=black_img

        # Locate right image first non zero column for blending
        c=0
        x=0
        y=int(img_right.shape[0]/2)
        while c == 0:
            x+=1
            if img_right[y,x]!=0:
                c=1

        left_seam=x-10
        # Locate left image first non zero column for blending
        c=0
        x=img_left.shape[1]
        y=int(img_left.shape[0]/2)
        while c==0:
            x-=1
            if img_left[y,x]!=0:
                c=1
        right_seam=x+10

        # Blend between first non-zero in right image (left seam) and left image original width (right seam)
        output=img_right.copy()
        w1=img_left.shape[1]
        h1=img_left.shape[0]

        output=cv2.GaussianBlur(output,(3,3),0)
        for j in range(0,w1):
            for k in range(0,h1):
                if j>left_seam and j<right_seam and img_left[k,j]!=0 and img_right[k,j]!=0:
                   output[k,j]= 0.9*((j-left_seam)/(right_seam-left_seam)*img_right[k,j]+(right_seam-j)/ \
							  (right_seam-left_seam)*img_left[k,j]) + 0.05*img_left[k,j]+0.05*img_right[k,j]
                else:
                    if img_left[k,j] == img_right[k,j]:
                        output[k,j]=img_left[k,j]
                    if img_left[k,j] != img_right[k,j]:

                        if img_left[k,j]!=0 and img_right[k,j]!=0:
                            output[k,j]=(img_left[k,j]+img_right[k,j])/2
                        elif img_left[k,j] == 0:
                            output[k,j]=img_right[k,j]
                        elif img_right[k,j] == 0:
                            output[k,j]=img_left[k,j]

        # Locate Output right edge for trimming
        c=0
        x=output.shape[1]
        y=int(output.shape[0]/2)
        while c==0:
            x-=1
            if output[y,x]!=0:
                c=1

        output=output[:,0:(x)]
        output=cv2.GaussianBlur(output,(3,3),0)

        cv2.imwrite(r'C:\Users\Matt\Desktop\Results\\'+file+'right'+str(i)+'.png',img_right)
        cv2.imwrite(r'C:\Users\Matt\Desktop\Results\\'+file+'left'+str(i)+'.png',img_left)
        cv2.imwrite(r'C:\Users\Matt\Desktop\Results\\'+file+'output'+str(i)+'.png',output)
        img_left=output

