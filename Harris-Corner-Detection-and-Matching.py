import cv2
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage.measurements import label

def slice_in_three(matrix):   
     b = matrix[:,:,0].astype(np.float)
     g = matrix[:,:,1].astype(np.float)
     r = matrix[:,:,2].astype(np.float)
     return b,g,r

def Differentiation(im,blue,green,red):
    [rows,cols,depth] = Im1.shape
    Ix = np.zeros([rows,cols]).astype(np.float)
    Iy = np.zeros([rows,cols]).astype(np.float)
    gx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix[:,:] = Ix[:,:] + np.abs(scipy.signal.convolve2d(blue,gx,mode='same'))
    Ix[:,:] = Ix[:,:] + np.abs(scipy.signal.convolve2d(green,gx,mode='same'))
    Ix[:,:] = Ix[:,:] + np.abs(scipy.signal.convolve2d(red,gx,mode='same'))
    Iy[:,:] = Ix[:,:] + np.abs(scipy.signal.convolve2d(blue,gy,mode='same'))
    Iy[:,:] = Ix[:,:] + np.abs(scipy.signal.convolve2d(green,gy,mode='same'))
    Iy[:,:] = Ix[:,:] + np.abs(scipy.signal.convolve2d(red,gy,mode='same'))  
    return Ix,Iy

Im1 = cv2.imread('im01.jpg')
Kernel_size  = 21
Kernel_sigma = 7
Blur1 = cv2.GaussianBlur(Im1,(Kernel_size,Kernel_size),Kernel_sigma)
B1,G1,R1 = slice_in_three(Blur1)
Im2 = cv2.imread('im02.jpg')
Kernel_size  = 21
Kernel_sigma = 7
Blur2 = cv2.GaussianBlur(Im2,(Kernel_size,Kernel_size),Kernel_sigma)
B2,G2,R2 = slice_in_three(Blur2)

Ix1,Iy1 = Differentiation(Im1,B1,G1,R1 )
Ix1[:,0]=Ix1[:,-1]=Ix1[0,:]=Ix1[-1,:]=0
Iy1[0,:]=Iy1[-1,:]=Iy1[:,0]=Iy1[:,-1]=0
Ixx1 = Ix1**2
Iyy1 = Iy1**2
Ixy1 = Ix1*Iy1
Gradian1 = (Ixx1+Iyy1)**(0.5)
cv2.imwrite('res01_grad.jpg',Gradian1)
Ix2,Iy2 = Differentiation(Im2,B2,G2,R2)
Ix2[:,0]=Ix2[:,-1]=Ix2[0,:]=Ix2[-1,:]=0
Iy2[0,:]=Iy2[-1,:]=Iy2[:,0]=Iy2[:,-1]=0
Ixx2 = Ix2**2
Iyy2 = Iy2**2
Ixy2 = Ix2*Iy2
Gradian2 = (Ixx2+Iyy2)**(0.5)
cv2.imwrite('res02_grad.jpg',Gradian2)

Kernel_size  = 21
Kernel_sigma = 7
Sxx1  = cv2.GaussianBlur(Ixx1,(Kernel_size,Kernel_size),Kernel_sigma)
Syy1  = cv2.GaussianBlur(Iyy1,(Kernel_size,Kernel_size),Kernel_sigma)
Sxy1  = cv2.GaussianBlur(Ixy1,(Kernel_size,Kernel_size),Kernel_sigma)

Sxx2  = cv2.GaussianBlur(Ixx2,(Kernel_size,Kernel_size),Kernel_sigma)
Syy2  = cv2.GaussianBlur(Iyy2,(Kernel_size,Kernel_size),Kernel_sigma)
Sxy2  = cv2.GaussianBlur(Ixy2,(Kernel_size,Kernel_size),Kernel_sigma)

Det1   = Sxx1*Syy1 - Sxy1**2
Det2   = Sxx2*Syy2 - Sxy2**2
Trace1 = Sxx1+Syy1
Trace2 = Sxx2+Syy2
K = 0.035
R1 = Det1- K * (Trace1**2)
cv2.imwrite('res03_score.jpg',R1)
R2 = Det2- K * (Trace2**2)
cv2.imwrite('res04_score.jpg',R2)

Threshold = 70
Threshold1 = (R1>Threshold) * 1
cv2.imwrite('res05_thresh.jpg',Threshold1* 255)
Threshold2 = (R2>Threshold) * 1
cv2.imwrite('res06_thresh.jpg',Threshold2* 255)

Structure = np.ones((3, 3), dtype=np.int)
Label1,Numofcomponents1 = label(Threshold1, Structure)
Indx1 = np.zeros([2,Numofcomponents1]).astype(np.int32)    
MaximumSuppression1 = np.zeros_like(R1)

Harris1 = np.zeros_like(Im1)
Harris2 = np.zeros_like(Im2)
for i in range(1,Numofcomponents1+1):
    CompIndx1 = np.array(np.where([Label1==i]))[1:3,:]
    Comp1 = R1[tuple(CompIndx1)]
    Maximum1 = np.array(np.where(Comp1==np.max(Comp1)))[0]
    MaximumIndx1 = CompIndx1[:,Maximum1]
    Indx1[:,i-1] =  [MaximumIndx1[0][0],MaximumIndx1[1][0]]
    MaximumSuppression1[tuple(Indx1[:,i-1])]=1
    Harris1 = cv2.circle(Harris1,(MaximumIndx1[1][0],MaximumIndx1[0][0]),3,(255,255,255),1)
cv2.imwrite('res07_harris.jpg',Harris1)
 
Label2,Numofcomponents2 = label(Threshold2, Structure)   
Indx2 = np.zeros([2,Numofcomponents2]).astype(np.int32)   
MaximumSuppression2 = np.zeros_like(R2)
for i in range(1,Numofcomponents2+1):
    CompIndx2 = np.array(np.where([Label2==i]))[1:3,:]
    Comp2 = R1[tuple(CompIndx2)]
    Maximum2 = np.array(np.where(Comp2==np.max(Comp2)))[0]
    MaximumIndx2 = CompIndx2[:,Maximum2]
    Indx2[:,i-1] =  [MaximumIndx2[0][0],MaximumIndx2[1][0]]
    MaximumSuppression2[tuple(Indx2[:,i-1])]=1 
    Harris2 = cv2.circle(Harris2,(MaximumIndx2[1][0],MaximumIndx2[0][0]),3,(255,255,255),1)       
cv2.imwrite('res08_harris.jpg',Harris2)   

N = 45
Des1 = np.zeros([Numofcomponents1,N**2,3])
for k in range(Numofcomponents1):
    for j in range(N):
        for i in range(N): 
            Des1[k,i+j*N,:] = Im1[Indx1[0,k]-3+j,Indx1[1,k]-3+i,:]
            
Des2 = np.zeros([Numofcomponents2,N**2,3])
for k in range(Numofcomponents2):
    for j in range(N):
        for i in range(N): 
            Des2[k,i+j*N,:] = Im2[Indx2[0,k]-3+j,Indx2[1,k]-3+i,:]
            
MatchFactor = np.zeros([Numofcomponents1,Numofcomponents2]) 
for j in range(Numofcomponents1):
    for i in range(Numofcomponents2):
        MatchFactor[j,i] = np.sum(np.abs(Des1[j,:,:]-Des2[i,:,:]))
BestMatch1 = np.zeros([Numofcomponents1,2,2]).astype(np.int32)
for i in range(Numofcomponents1):
    BestMatch1[i,0,0] = np.min(MatchFactor[i,:])
    BestMatch1[i,0,1] = np.array(np.where(MatchFactor[i,:]==np.min(MatchFactor[i,:])))[0][0]
    MatchFactor[i,BestMatch1[i,0,1]] = np.max(MatchFactor[i,:])
    BestMatch1[i,1,0] = np.min(MatchFactor[i,:])  
    BestMatch1[i,1,1] = np.array(np.where(MatchFactor[i,:]==np.min(MatchFactor[i,:])))[0][0]   
    MatchFactor[i,BestMatch1[i,0,1]] = BestMatch1[i,0,0]  
    
BestMatch1[:,0,0] = BestMatch1[:,0,0] + (BestMatch1[:,0,0]==0)*1
DistanceRatio1 = BestMatch1[:,1,0]/BestMatch1[:,0,0]
RatioThreshld = 1.2
Pair1 = DistanceRatio1 > RatioThreshld

BestMatch2 = np.zeros([Numofcomponents2,2,2]).astype(np.int32)
for i in range(Numofcomponents2):
    BestMatch2[i,0,0] = np.min(MatchFactor[:,i])
    BestMatch2[i,0,1] = np.array(np.where(MatchFactor[:,i]==np.min(MatchFactor[:,i])))[0][0]
    MatchFactor[BestMatch2[i,0,1],i] = np.max(MatchFactor[:,i])
    BestMatch2[i,1,0] = np.min(MatchFactor[:,i])  
    BestMatch2[i,1,1] = np.array(np.where(MatchFactor[:,i]==np.min(MatchFactor[:,i])))[0][0]   
    MatchFactor[BestMatch2[i,0,1],i] = BestMatch2[i,0,0]     
BestMatch2[:,0,0] = BestMatch2[:,0,0] + (BestMatch2[:,0,0]==0)*1
DistanceRatio2 = BestMatch2[:,1,0]/BestMatch2[:,0,0]   
Pair2 = DistanceRatio2 > RatioThreshld          

PairList1 = []
PairList2 = []
for i in range(Numofcomponents1):
    if Pair1[i]:
        if Pair2[BestMatch1[i,0,1]]:
            if BestMatch2[BestMatch1[i,0,1],0,1] == i :
                PairList1.append([Indx1[0,i],Indx1[1,i]])
                PairList2.append([Indx2[0,BestMatch1[i,0,1]],Indx2[1,BestMatch1[i,0,1]]])
                
PairList1 = np.array(PairList1)
PairList2 = np.array(PairList2) 
Corres1 = Im1.copy() 
Corres2 = Im2.copy()         
for i in range(np.size(PairList1,axis=0)):
    Corres1 = cv2.circle(Corres1,(PairList1[i][1],PairList1[i][0]),5,(255,0,0),3)
    Corres2 = cv2.circle(Corres2,(PairList2[i][1],PairList2[i][0]),5,(255,0,0),3)
    
cv2.imwrite('res09_corres.jpg',Corres1)
cv2.imwrite('res10_corres.jpg',Corres2)        

rows,cols,dim = Im1.shape
Corres  = np.zeros([rows,cols*2,3])
Corres  [:,0:cols,:] = Corres1
Corres  [:,cols:,:]  = Corres2

for i in range(10):
        Corres = cv2.line(Corres,(PairList1[i][1],PairList1[i][0]),(PairList2[i][1]+cols,PairList2[i][0]),(255,0,0),1)  

cv2.imwrite('res11.jpg',Corres) 