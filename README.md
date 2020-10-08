# Description 
This repository contains the program for Optical flow and Global motion estimation in the image plane with RANSAC algorithm. Further Measures the information content derived from 
the video sequences with different metrics such as Mean squared error (MSE), Entropy and Peak signal to noise ratio (PSNR). You can also inspect the Optical flow vectors, 
Error images (Frame Difference image) , original frame after global motion compensation.

- Hyperparameter : DeltaT <integer> (Difference between the frame of measurement)

# How to run 
- Python run_estimation.py -video file path- -DeltaT- 
