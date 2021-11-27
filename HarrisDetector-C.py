#!/usr/bin/env python
# coding: utf-8

#Class:  Advanced image processing 
#TP1: Detection of Region Of Interest 
#Master TSI - IMOVI, 2021-2022
#Professor: M.M. Nawaf
#Developper : Assia CHAHIDI
# 
# 
# Objective : In this TP, we will get to impelemnt a different methods to detect the corners in a image. Also, understand the importance of the different steps to implement the Harris algorithm. In this TP we have the right to use just OpenCv, numpy and matplotlib libraries. 

# In[1]:calculate the intensity of the image


#Libraries allow to implement 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

#Read the image 
img = cv2.imread('chessboard00.png')
# RGB to to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# Convert the array type from float64 to flot32
img = np.float32(img)

#Define two kernels for the convolution 
ab = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)*0.5
ba = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], np.float32)*0.5

#The convolution between the kernels previously defined and the the grayscale image
Ix = cv2.filter2D(img, cv2.CV_32F, ab)
Iy = cv2.filter2D(img, cv2.CV_32F, ba)

#plotting 

fig, axs = plt.subplots(1,3, figsize=(20, 5))
axs[0].set_title('Original image')
axs[0].imshow(img, cmap='gray')

axs[1].set_title('Ix image')
axs[1].imshow(Ix, cmap='gray')

axs[2].set_title('Iy image')
axs[2].imshow(Iy, cmap='gray')
plt.show()


# In[2]:Convolution of the square product with a gaussian filter


#Product of Ix
Ix_sq = Ix**2
#Product of Iy
Iy_sq = Iy**2
#Product of matrix Ix and Iy
Ixy = Ix*Iy
#Apply the Gaussian filter of the previous computed matrix with a kernel size (9,9) and sigma equal to 2.  
GaussianIx_square = cv2.GaussianBlur(Ix_sq,(9,9),2)
GaussianIy_square = cv2.GaussianBlur(Iy_sq,(9,9),2)
GaussianIxy = cv2.GaussianBlur(Ixy,(9,9),2)
#plotting

fig, axs = plt.subplots(1,3, figsize=(20, 5))
axs[0].set_title('Ix_square')
axs[0].imshow(GaussianIx_square, cmap='gray')

axs[1].set_title('Iy_square')
axs[1].imshow(GaussianIy_square, cmap='gray')

axs[2].set_title('Ixy')
axs[2].imshow(GaussianIxy, cmap='gray')
plt.show()


# In[4]:Implementing  the Harris and shi-tomasi algorithms


#function to find the local maximum 
def localMax(p,i,j) :  
   # if i > 0 and j > 0 and i < p.shape[0]-1 and j < p.shape[1]-1 : 
        return ( p[i][j] > np.array([p[i-1][j],p[i][j-1],p[i-1][j-1],p[i+1][j],p[i][j+1],p[i+1][j+1],p[i-1][j+1],p[i+1][j-1]]).max() )#.min()


# In[5]:


k = 0.04
n_h= 253 # height = img.shape[0]
n_w = 250 #width = img.shape[1]
window = 3 #window size
offset = 1 #the window will shift by 1 

#define the matrices
Sxx = np.zeros((n_h,n_w))
Sxy = np.zeros((n_h,n_w))
Syy = np.zeros((n_h,n_w))
detM = np.zeros((n_h,n_w))
trM = np.zeros((n_h,n_w))
E = np.zeros((n_h,n_w)) 
R = np.zeros((n_h,n_w)) 
M = np.zeros((n_h,n_w))
Max = np.zeros((n_h,n_w))
max = 0
result = np.zeros((n_h,n_w))

#loop to find R and E 
for y in range(offset, n_h-offset):
    for x in range(offset, n_w-offset):
        #loop the window through the gaussian filter of the previous computed matrices: Ix_square, Iy_square, Ixy_square. 
        windowIxx = GaussianIx_square [y-offset:y+offset+1, x-offset:x+offset+1]
        windowIxy = GaussianIxy[y-offset:y+offset+1, x-offset:x+offset+1]
        windowIyy = GaussianIy_square[y-offset:y+offset+1, x-offset:x+offset+1]
        #Sum all the windows
        Sxx[y,x] = windowIxx.sum()
        Sxy[y,x] = windowIxy.sum()
        Syy[y,x] = windowIyy.sum()
        #computing the derivative of matrix M = (Ix_square * Iy_square) - (Ixy_square²)
        detM[y,x] = (Sxx[y,x] * Syy[y,x]) - (Sxy[y,x]**2)
        #trace
        trM[y,x] = Sxx[y,x] + Syy[y,x]
        #R = det(M) -k*trace(M)²
        R[y,x] = detM[y,x] - k*(trM[y,x]**2)
        #E = det(M)/trace(M)
        E[y,x] = detM[y,x]/trM[y,x]
        #Finding the maximum value in matrix R
        if R[y,x] > max:
            max = R[y,x]


# In[6]:


#plotting

fig, axs = plt.subplots(1,2, figsize=(20, 5))
axs[0].set_title('Harris method')
axs[0].imshow(R, cmap='gray')
axs[1].set_title('Shi-Tomasi')
axs[1].imshow(E, cmap='gray')
plt.show()

###Remark:
#We can observe that the shi-tomasi did a better job in pixel detection than harris algorithm. 
# In[7]:


#Corners finding for Harris matrix
for y in range(n_h-offset) : 
    for x in range(n_w-offset) : 
        #putting a threshold for R using the local maximum function
        if R[y,x] > 0.04*max  and localMax(R,y,x) : 
            result[y,x] = 1 

a, b = np.where(result == 1)
#plotting
plt.figure()
plt.plot(b, a, 'r+')
plt.imshow(img, 'gray')
plt.title("Corners detection on Harris method")

# In[8]:

#corners finding for E matrix
for y in range(n_h-offset) : 
    for x in range(n_w-offset) :
        #using local maximum to find the cordinates in matrix E 
        if E[y,x] > 0.04*max  and localMax(E,y,x) : 
            result[y,x] = 1 
#plotting
a, b = np.where(result == 1)
plt.figure()
plt.plot(b, a, 'r+')
plt.imshow(img, 'gray')
plt.title("Corners detection on Shi-Tomasi method")
#Remark
#Yes, it's the results that we are looking for
# In[9]:Compute the non maximal supression on Harris and shi-tomasi


#Function to compute the non maximal supression
def nonMaximalSupress2(image,NHoodSize):
    #
    dX, dY = NHoodSize#the window 
    M, N = image.shape#original image shape
    #loop the window on the image to find the non maximum values
    for x in range(0,M-dX+1):
        for y in range(0,N-dY+1):
            window = image[x:x+dX, y:y+dY]
            if np.sum(window)==0:
                localMax=0
            else:
                localMax = np.amax(window)
            maxCoord = np.argmax(window)
            # zero all but the localMax in the window
            window[:] = 0
            window.flat[maxCoord] = localMax
    return image


# In[10]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt

#Apply the function on our original image
img = cv2.imread('chessboard00.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## RGB to to gray scale
img = np.float32(img)
nonMaximalSupress2(image=R,NHoodSize= np.array((11,11)))
nonMaximalSupress2(image=E,NHoodSize= np.array((11,11)))


# In[11]:

#plotting
fig, axs = plt.subplots(1,2, figsize=(20, 5))
axs[0].set_title('Non maximal supression on Harris method')
axs[0].imshow(R, cmap='gray')
axs[1].set_title('Non maximal supression on Shi-tomasi method')
axs[1].imshow( E, cmap='gray')
plt.show()

#Remark
#using the two methods, we are achieving the same goal of corners detection. 
# In[ ]:Detect the sous pixel values for Harris and shi-tomasi algorithms

#Find the sous pixel values

filename = 'chessboard00.png'
imgR = cv2.imread(filename)
imgE = cv2.imread(filename)
grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
grayE = cv2.cvtColor(imgE,cv2.COLOR_BGR2GRAY)
dstR = cv2.dilate(R,None)#morphological operation apply on R
ret, dstR = cv2.threshold(dstR,0.01*dstR.max(),255,0)#threshold to select only the maximum pixels
dstR = np.uint8(dstR)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dstR)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
cornersR = cv2.cornerSubPix(grayR,np.float32(centroids),(5,5),(-1,-1),criteria)
# Now draw them
res = np.hstack((centroids,cornersR))
resR = np.int0(res)
imgR[resR[:,1],resR[:,0]]=[0,0,255]
imgR[resR[:,3],resR[:,2]] = [0,255,0]

#Apply on E
dstE = cv2.dilate(E,None)#morphological operation 
ret, dstE = cv2.threshold(dstE,0.01*dstE.max(),255,0)#threshold to select only the maximum pixels
dstE = np.uint8(dstE)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dstE)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
cornersE = cv2.cornerSubPix(grayE,np.float32(centroids),(5,5),(-1,-1),criteria)
# Now draw them
res = np.hstack((centroids,cornersE))
resE = np.int0(res)
imgE[resE[:,1],resE[:,0]]=[0,0,255]
imgE[resE[:,3],resE[:,2]] = [0,255,0]

#write the output
#plotting
fig, axs = plt.subplots(1,2, figsize=(20, 5))
axs[0].set_title("Sous pixel of Harris method")
axs[0].imshow(imgR, cmap='gray')
axs[1].set_title("Sous pixel of Shi-tomasi method")
axs[1].imshow( imgE, cmap='gray')
plt.show()

