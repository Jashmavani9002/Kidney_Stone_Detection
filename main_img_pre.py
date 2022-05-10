import cv2
import numpy as np

def Watershed(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image',gray_image)
    # img = img.astype("uint8")
    ret, thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Thresh',thresh)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow('opening',opening)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # cv2.imshow('BG',sure_bg)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # cv2.imshow('dist_transform',dist_transform)
    ret, sure_fg = cv2.threshold(dist_transform,0.14*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    # cv2.imshow('FG',sure_fg)


    unknown = cv2.subtract(sure_bg,sure_fg)
    # cv2.imshow('UK',unknown)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    img2 = cv2.imread('test3.jpeg',1)
    img2 = cv2.medianBlur(img2,5)
    gamma_corrected = np.array(255*(img2 / 255) ** 2.5, dtype = 'uint8')
    # markers = cv2.integral(markers)

    # cv2.imshow('Markers',markers)
    print(type(markers))
    markers = cv2.watershed(gamma_corrected,markers)
    cv2.imwrite('marker_img.jpg', markers)
    marker_img = cv2.imread('marker_img.jpg',1)
    cv2.imshow("image", marker_img)
    gamma_corrected[markers == -1] = [0,0,255]

    return gamma_corrected

 # Salt Paper noise remove :  Median Filter
def filter_median(image) :
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image',gray_image)
    img_median = cv2.medianBlur(gray_image, 5)
    return img_median

# noise remove :  Median Filter
def filter_gaussian(image) :
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image',gray_image)
    blur = cv2.GaussianBlur(gray_image,(3,3),0)

# Apply Laplacian operator in some higher datatype
    img_guassian = cv2.Laplacian(blur,cv2.CV_64F)
    return img_guassian

def Laplacian(img,par):  
    lap = cv2.Laplacian(img,cv2.CV_64F)
    sharp = img - par*lap
    sharp = np.uint8(cv2.normalize(sharp, None, 0 , 255, cv2.NORM_MINMAX))
    return sharp

image = cv2.imread('test3.jpeg')

output = Watershed(image)

cv2.imshow('Output',output)

cv2.waitKey(0)
