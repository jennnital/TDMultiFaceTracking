# TouchDesigner Multi-face Tracking with OpenCV and YuNet

## Overview  
This repository provides implementations for multi-face tracking in TouchDesigner using two methods:  
- **YuNet**: A deep learning-based face detection model.  
- **OpenCV Cascade Classifier**: A traditional method using Haar cascades.  

---

## YuNet  
In the folder `face_detection_yunet`, you will find the TouchDesigner implementation of YuNet.

### Video Source  
- This example uses `op('VideoSource')` as the source for face detection.  
- To test with your own video, replace the file path in the operator.  

### Model Setup  
- In `DAT script2_callbacks`, update the system path to the `face_detection_yunet` folder on **line 6**.  
- Path format is `r"C:\....` for Windows.  

### Hardware Acceleration  
- This implementation attempts to use **CUDA** if OpenCV DNN CUDA support is available.  
- If CUDA is not found, it will automatically switch to **CPU** processing.  

---

## OpenCV  
The folder `OpenCV` contains an implementation using the **Cascade Classifier** from OpenCV in TouchDesigner.  

### Features  
- This example detects faces and applies a blur effect on them.  

---

## Requirements  
- TouchDesigner  
- OpenCV with DNN module support (for YuNet)  
- CUDA (optional, for hardware acceleration)  

---

## Installation  
1. Clone this repository:  
    ```bash
    git clone https://github.com/yourusername/TD-Multi-face-Tracking.git
    ```
2. Install required Python packages:  
    ```bash
    pip install opencv-python opencv-python-headless
    ```
3. Update system paths in the scripts as mentioned above.  

---

## Usage  
1. Open the corresponding TouchDesigner project file.  
2. Adjust video source paths as needed.  
3. Run the project and observe real-time face tracking.  

---

## License  
This project is licensed under the MIT License.  

---

## Acknowledgments  
- [OpenCV](https://opencv.org/)  
- [YuNet Face Detection Model](https://github.com/opencv/opencv_zoo)  
