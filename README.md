# Machine-Vision-GUI
This repository includes ResNet, YoloV8, U-Net, Padim(Bottle) Style Transfer (Van Gogh), Filtering and Color Space, Edge and Corner Detection (Sobel, Canny, Laplacian, Prewitt, Shi-Tomasi, FAST, ORB), Morphological methods, Line and Circle Detection via Hough, Geometric Transformations examples 

This project is a PyQt6-based graphical user interface (GUI) for advanced image processing.
It combines classical image filters with state-of-the-art deep learning models such as YOLOv8, U-Net, PaDiM anomaly detection, and Neural Style Transfer into a single interactive application.

Project Structure
23568032/
│── 23568032.py           # Main GUI (PyQt6)
│── inference_padim.py    # PaDiM loading & inference
│── unet.py, unet_parts.py # U-Net model implementation
│── yolov8_video.py       # YOLOv8 on video input
│── yolov8_webcam.py      # YOLOv8 on webcam input
│── style_transfer/       # Neural style transfer models
│── MVtec/                # MVTEC dataset (for anomaly detection)
│── results/              # Output results
│── imagenet_classes.txt  # ImageNet class labels
│── padim_bottle.pth      # PaDiM pretrained weights
│── yolov8n.pt            # YOLOv8 pretrained model
│── unet_carvana_cpu.pth  # U-Net pretrained weights

pip install -r requirements.txt
