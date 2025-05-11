# Digital Image Processing Laboratory

## Overview

This repository contains MATLAB implementations for various digital image processing techniques, developed as part of the "Digital Image Processing and Analysis" course at the Computer Engineering & Informatics Department, University of Patras (Spring Semester 2024-25).

The implementations cover fundamental techniques for image enhancement, restoration, compression, and feature extraction. Each exercise demonstrates both the theoretical foundations and practical applications of digital image processing algorithms.

## Exercises

### Exercise 1: Frequency-Domain Filtering

Implementation of image filtering in the frequency domain using the Discrete Fourier Transform (DFT).

**Key features:**
- Preprocessing and linear stretching of pixel values
- 2D DFT implemented through row-column decomposition using 1D FFT
- Low-pass filtering in the frequency domain
- Inverse DFT to restore the filtered image
- Visualization of spectra in both linear and logarithmic scales

**Sample image:** `moon.jpg`

### Exercise 2: Image Compression using DCT

Implementation of image compression techniques using the Discrete Cosine Transform (DCT).

**Key features:**
- Division of the image into non-overlapping blocks (32×32 pixels)
- Application of 2D-DCT to each block
- Coefficient selection using two methods:
  - Zone method (selecting coefficients based on frequency region)
  - Threshold method (selecting coefficients based on magnitude)
- Compression ratio control through percentage parameter
- Analysis of mean squared error (MSE) vs. compression ratio

**Sample image:** `board.png`

### Exercise 3: Noise Filtering

Implementation of spatial domain filters for noise removal from images.

**Key features:**
- Gaussian noise addition with controlled SNR (15 dB)
- Impulse noise (salt & pepper) addition at 20% density
- Mean filter (moving average) implementation
- Median filter implementation
- Comparative analysis of filter performance for different noise types
- Sequential application of filters for combined noise

**Sample image:** `tiger.mat`

### Exercise 4: Histogram Equalization

Implementation of histogram equalization techniques for enhancing dark images.

**Key features:**
- Calculation and visualization of image histograms
- Global histogram equalization using `histeq`
- Local (adaptive) histogram equalization using `adapthisteq` (CLAHE)
- Analysis of enhancement effectiveness for night-time road images

**Sample images:** `dark_road_1.jpg`, `dark_road_2.jpg`, `dark_road_3.jpg`

### Exercise 5: Image Restoration and Deconvolution

Implementation of image restoration techniques for denoising and deblurring.

**Part A: Wiener Filtering for Noise Removal**
- Addition of Gaussian noise with SNR = 10 dB
- Wiener filtering with known noise-to-signal ratio
- Adaptive Wiener filtering with unknown noise characteristics

**Part B: Inverse Filtering for Deblurring**
- Estimation of Point Spread Function (PSF)
- Inverse filtering with threshold to prevent noise amplification
- Analysis of MSE vs. threshold relationship

**Sample image:** `new_york.png`

### Exercise 6: Edge Detection and Hough Transform

Implementation of edge detection and line detection techniques.

**Key features:**
- Sobel edge detection with gradient magnitude calculation
- Global thresholding using Otsu's method
- Canny edge detection for comparison
- Hough transform for line detection
- Categorization of lines by orientation (vertical, horizontal, diagonal)
- High-resolution visualization of detected lines

**Sample image:** `hallway.png`

## Requirements

- MATLAB R2020b or newer
- Image Processing Toolbox
- Signal Processing Toolbox

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/digital-image-processing-lab.git
   ```

2. Navigate to the exercise directory of interest

3. Run the MATLAB script:
   ```matlab
   run askisiX.m  % Where X is the exercise number (1-6)
   ```

4. Modify parameters as needed to experiment with different settings

## File Structure

```
├── potamianos-1084537-report.pdf
├── Images/
│   ├── Ασκηση 1/
│   │   └── moon.jpg
│   ├── Ασκηση 2/
│   │   └── board.png
│   ├── Ασκηση 3/
│   │   └── tiger.mat
│   ├── Ασκηση 4/
│   │   ├── dark_road_1.jpg
│   │   ├── dark_road_2.jpg
│   │   └── dark_road_3.jpg
│   ├── Ασκηση 5/
│   │   ├── new_york.png
│   │   └── psf.p
│   └── Ασκηση 6/
│       └── hallway.png
├── askisi1.m
├── askisi1.mlx
├── askisi1.ipynb
├── askisi2.m
├── askisi2.mlx
├── askisi2.ipynb
├── askisi3.m
├── askisi3.mlx
├── askisi3.ipynb
├── askisi4.m
├── askisi4.mlx
├── askisi4.ipynb
├── askisi5.m
├── askisi5.mlx
├── askisi5.ipynb
├── askisi6.m
├── askisi6.mlx
├── askisi6.ipynb
└── README.md
```

## Results

The scripts generate visualization figures that illustrate the effects of different processing techniques. These results can be saved by setting the appropriate flag in each script. Example outputs include:

- Frequency spectra and filtered images
- Compressed images at different quality levels
- Noise-filtered images using different filter types
- Histogram-equalized images
- Restored images after deblurring
- Edge and line detection results

## Notes

- Each script contains detailed comments explaining the theoretical background and implementation details
- Parameters can be adjusted at the beginning of each script to experiment with different settings
- The code is structured to facilitate understanding of the underlying algorithms rather than optimizing for computational efficiency

## Author

ANGELOS NIKOLAOS POTAMIANOS

Department of Computer Engineering & Informatics  
University of Patras  
Academic Year 2024-25

## License

This project is licensed under the MIT License - see the LICENSE file for details.