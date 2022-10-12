# Convolution
Implementing Convolutions with Python:-



Convolution is simply the sum of element-wise matrix multiplication
between the kernel and neighborhood that the kernel covers of the input image.



Kernel:

We use an odd kernel size to ensure there is a valid integer (x;y)-coordinate at the center of the image. 
On the left, we have a 3X3 matrix.  The center of the matrix is located at x = 1;y = 1 
where the top-left corner of the matrix is used as the origin and our coordinates are zero-indexed. 

But on the right, we have a 2X2 matrix. The center of this matrix would be located at x = 0:5;y = 0:5.
But as we know, without applying interpolation, there is no such thing as pixel location (0:5;0:5) – our pixel coordinates must be integers! 

This reasoning is exactly why we use odd kernel sizes: to always ensure there is a valid (x;y)-coordinate at the center of the kernel.


 ![GIT ](https://user-images.githubusercontent.com/96985326/195270445-2786c77c-c8af-4d9e-9c6c-848a9ecbf7df.jpg)




This are the kernel which we had used in our code

Sharpen kernel responsible for sharpening an image.

Laplacian kernel used to detect edge-like regions.

Sobel kernels can be used to detect edge-like regions along both the x and y axis respectively.

sobelX kernel is used to find vertical edges in the image, while the sobelY kernel reveals horizontal edges.
