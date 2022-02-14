import cv2
import numpy as np

path = "docs/doc1.jpg"
imgWt, imgHt = 640, 480



def preprocess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    # Hilight Edges
    # return imgCanny
    kernel = np.ones((5, 5))
    # We apply Two iteration of dialation (to make img thicker) 
    # and one iteration of ersoion to make thinner again 
    # Helps to get better judgement of edges
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)
    imgThres =imgErode
    return imgThres

def getContours(img):
    contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #return corners of biggest closed space i.e. document
    maxArea, biggest = 0, 0
    for cnt in contour:
        area = cv2.contourArea(cnt)
        # print(area)
        if(area>5000 and area> maxArea):
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 1)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
            objCor = len(approx)
            if(objCor == 4):
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 255), 12)
    return biggest

def order (myPoints):
   myPoints = myPoints.reshape( (4,2)) 
   myPointsNew = np.zeros ((4,1,2),np.int32) 
   add = myPoints.sum(1)
   print("add", add)
   myPointsNew[0] = myPoints[np.argmin(add)]
   myPointsNew[3] = myPoints[np.argmax(add)]
   diff = np.diff(myPoints,axis=1)
   myPointsNew[1]= myPoints[np.argmin(diff)]
   myPointsNew[2] = myPoints[np.argmax(diff)]
   print("NewPoints",myPointsNew)
   return myPointsNew
                  


def getWarp(img, biggest):
    biggest = order(biggest)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [imgWt, 0], [0, imgHt], [imgWt, imgHt]])
    matrix= cv2.getPerspectiveTransform(pt1, pt2)
    imgOut = cv2.warpPerspective(img, matrix, (imgWt, imgHt))
    return imgOut



# From Particular image in "path"

img = cv2.imread(path)
img = cv2.resize(img, (imgWt, imgHt))
imgContour = img.copy()
imgPre= preprocess(img)
biggest = getContours(imgPre)
imgWarp = getWarp(img, biggest)
print(biggest)
# cv2.imshow("imgPre", imgPre)
# cv2.imshow("imgCont", imgContour)
final_img = np.hstack((img, imgWarp))
# cv2.imshow("IMAGe", img)
cv2.imshow("imgWarp", final_img)
cv2.waitKey(0)



# ********************************************************************************************
#  FROM CAMERA
    # COMMENT OUT THE CODE FROM (- From Particular image in "path")
    # UNCOMMENT THE CODE BELOW FOR SCANNING DOC FROM CAMERA


        # 0: For Scanning with primary Camera of device
        # 1: Secondary Camera installed on device 
# cap = cv2.VideoCapture(0)
# cap.set(3, imgWt)
# cap.set(4, imgHt)
# cap.set(10, 1)

# while(1):
#     flag, img = cap.read()
#     img = cv2.resize(img, (imgWt, imgHt))
#     imgFinal = img.copy()

#     imgContour = img.copy()
#     imgPre= preprocess(img)
#     biggest = getContours(imgPre)
#     imgWarp = getWarp(img, biggest)
#     print(biggest)
#     # cv2.imshow("imgPre", imgPre)
#     # cv2.imshow("imgCont", imgContour)
#     final_img = np.hstack((img, imgWarp))
#     # cv2.imshow("IMAGe", img)
#     cv2.imshow("imgWarp", final_img)
    
    
#     # cv2.imshow("cam", img)
#     key = cv2.waitKey(24)
#     if(key%256 == 27):
#         break