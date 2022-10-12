#importing necessary libraries
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import argparse

# Convolution is simply the sum of element-wise matrix multiplication
# between the kernel and neighborhood that the kernel covers of the input image.

def convolve (image, kernel):
    #grap the spatial dimension of the image and kernel
    
    (img_h, img_w) = image.shape[:2]
    (ker_h, ker_w) = kernel.shape[:2]
    
    # allocate memory for the output image, takerneling care to "pad"
    # the borders of the input image so the spatial size (i.e.,
    # width and height) are not reduced
    pad = (ker_w-1)//2
    
    # we want our output image to have the same dimensions as our input
    # image. To ensure the dimensions are the same, we apply padding 

    image = cv2.copyMakeBorder(image, pad,pad,pad,pad, cv2.BORDER_REPLICATE)
    output = np.zeros((img_h,img_w), dtype = 'float')
    
    #loop over the inputimage, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top-to-bottom
    for y in np.arange(pad, img_h + pad):
        for x in np.arange(pad, img_w + pad):
             #extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            
            #perform the actual convolution by taking the 
            #element wise multiplication between the roi and 
            #the kernel then summing the matrix
            
            k = (roi * kernel).sum()
            
            #store the convolved value in the output (x,y)
            #coordinate of the output image
            output[y - pad, x - pad] =  k
            
            #rescale_intensity function of scikit image
    # rescale the output image to be in the range [0, 255]
            
    output = rescale_intensity(output, in_range= (0,255))
    output = (output*255).astype('uint8')
            
            #return the output image
    return output
            
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image', required = True, help = 'path to the input image')
args = vars(ap.parse_args())

#construct average blurring kernels used to smooth an image
smallBlur = np.ones((7,7), dtype = 'float')* (1.0/(7*7))
largeBlur = np.ones((21,21), dtype = 'float')*(1.0/(21*21))

# To convince yourself that this kernel is performing blurring, notice how each entry in the kernel
# is an average of 1/S where S is the total number of entries in the matrix. 
# Thus, this kernel will multiply each input pixel by a small fraction and take the sum – this is exactly the definition of the average.
#We then have a kernel responsible for sharpening an image:

#construct a sharpening filer
sharpen = np.array((
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]), dtype = 'int')


#construct the laplacian kernel used to detect edge like regions of an image
laplacian = np.array((
    [0,1,0],
    [1,-4,1],
    [0,1,0]), dtype = 'int')
            
#construct the sobel kernel used to detect edge like regions along
#both the xand y axis respectively

#construct the sobel x asix kernel
sobelX = np.array((
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]), dtype = 'int')

#construct the sobel y axis kernel
sobelY = np.array((
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]), dtype = 'int')


#construct an emboss kernel

emboss = np.array((
    [-2,-1,0],
    [-1,1,1],
    [0,1,2]), dtype = 'int')


#construct a kernel bank  alist of kernels we're going to apply
#using both our custom 'convole' function and OPenCv's 'filter2D' function

kernelBank = (
    ('Small_blur',smallBlur),
    ('large_blur',largeBlur),
    ('sharpen',sharpen),
    ('laplacian',laplacian),
    ('sobel_x',sobelX),
    ('sobel_y', sobelY),
    ('emboss',emboss)
)

#load the input image and convert it to grayscale
image = cv2.imread(args['image'])
window_name = 'Original Image'
cv2.imshow(window_name, image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#loop over the kernels

for (kernelName , kernel) in kernelBank:
    #apply the kernel to the grayscalle image using both our cutsom
    #'convolve' functions and OPenCv's 'filter2D' function
    
    print('[INFO] applying {} kernel'.format(kernelName))
    convolveOutput = convolve(gray, kernel)
    opencvOutput = cv2.filter2D(gray, -1, kernel)
    
    #show the output images
    cv2.imshow('Originnal',gray)
    cv2.imshow('{} - convole'. format(kernelName), convolveOutput)
    cv2.imshow('{} -opencv'.format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



