import cv2 as cv
import os
#"D:\images\training\s_balti"
j=1
listing = os.listdir("F:/Hand Sign/dataset/raw data/C sign/")
for file in listing:
    print(file)
    img = cv.imread("F:/Hand Sign/dataset/raw data/C sign/"+ file)
    img = cv.resize(img,(200,200))
    cv.imwrite('F:/Hand Sign/dataset/Preprocess data/C sign/'+'C'+str(j)+'.png',img)
    print('Done with '+str(j)+' images (-_-)')
    j=j+1
