# Convolution
Implementing Convolutions with Python:-
To help us further understand the concept of convolutions, let’s look at some actual code that will
reveal how kernels and convolutions are implemented. This source code will not only help you
understand how to apply convolutions to images, but also enable you to understand what’s going on
under the hood when training CNNs.

Convolution is simply the sum of element-wise matrix multiplication
between the kernel and neighborhood that the kernel covers of the input image.

We use an odd kernel size to ensure there is a valid integer (x;y)-coordinate at the center of
the image (Figure 11.2). On the left, we have a 3X3 matrix. 
The center of the matrix is located at x = 1;y = 1 where the top-left corner of the matrix is used as the origin and our coordinates are
zero-indexed. 
But on the right, we have a 2X2 matrix. The center of this matrix would be located at x = 0:5;y = 0:5.But as we know, without applying interpolation, 
there is no such thing as pixel location (0:5;0:5) – our pixel coordinates must be integers! 
This reasoning is exactly why we use odd kernel sizes: to always ensure there is a valid (x;y)-coordinate at the center of the kernel.

