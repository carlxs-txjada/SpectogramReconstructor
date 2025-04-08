import numpy as np

# Input & Output folders
InputFolder = "Input"
OutputFolder = "Output"

# Visualization parameter
FigureSize = (10, 4)
ColorMapFormat = '%+2.0f dB'
SpectrogramTitle = "Original Spectrogram"

# Image filters
GaussianKernelSize = (5, 5)
SharpenKernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

CannyThreshold1 = 50
CannyThreshold2 = 150
