import sys, cv2, os, torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QFileDialog, QCheckBox, QComboBox, QScrollArea,
                             QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
                             QStatusBar, QMessageBox, QSlider, QSizePolicy, QLineEdit)
from PyQt6.QtWidgets import QStackedLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from inference_padim import load_model, preprocess_image, infer
from PIL import Image
from style_transfer.models.transformer_net import TransformerNet

class ProcessingControlDialog(QDialog):
    def __init__(self, parent, active_filters, callback, process_callback):
        super().__init__(parent)
        self.setWindowTitle("Active Processing Controls")
        self.setGeometry(300, 200, 400, 300)
        self.callback = callback  # Ana pencereye işlemleri geri bildirmek için
        self.process_callback = process_callback  # İşleme tekrar başlatmak için

        self.layout = QVBoxLayout()
        self.checkboxes = {}

        for filter_name, is_active in active_filters.items():
            if is_active:
                checkbox = QCheckBox(filter_name)
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(self.toggle_filter)
                self.layout.addWidget(checkbox)
                self.checkboxes[filter_name] = checkbox

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.layout.addWidget(self.close_button)

        self.setLayout(self.layout)

    def toggle_filter(self):
        updated_filters = {
            name: checkbox.isChecked()
            for name, checkbox in self.checkboxes.items()
        }
        self.callback(updated_filters)  # sadece disable listesi güncelle


class ImageProcessingGUI(QMainWindow):
    def is_image_loaded(self):
        return self.current_image is not None and self.processed_image is not None
    def __init__(self):
        super().__init__()

        # Tanımlamaları önce yap
        self.current_image = None
        self.processed_image = None
        self.processing_history = []
        self.video_capture = None
        self.original_width = None
        self.original_height = None
        self.active_filters = {
            "Gaussian Blur": False,
            "Median Filter": False,
            "Bilateral Filter": False,
            "Sobel Edge Detection": False,
            "Canny Edge Detection": False
        }
        self.disabled_filters = set()
        self.active_operations = {}
        self.perspective_src_points = []
        self.epipolar_left_image = None
        self.epipolar_right_image = None
        self.epipolar_input_left_label = QLabel("Left Input")
        self.epipolar_input_right_label = QLabel("Right Input")
        self.epipolar_output_label = QLabel("Output for 2 Input")
        for label in [self.epipolar_input_left_label, self.epipolar_input_right_label, self.epipolar_output_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(200, 200)
        # Deep learning için model ve transform
        self.classification_model = None
        self.classification_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_inference_active = False
        self.confidence_threshold = 0.5
        self.segmentation_model = None
        self.segmentation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((572, 572)),  # U-Net için yaygın giriş boyutu
            transforms.ToTensor(),
        ])
        self.segmentation_threshold = 0.5
        self.segmentation_inference_active = False
        self.padim_model = load_model("padim_bottle.pth")
        # ImageNet sınıf isimlerini yükle
        try:
            with open("imagenet_classes.txt", "r") as f:
                self.imagenet_classes = [line.strip() for line in f.readlines()]
        except Exception:
            self.imagenet_classes = [f"class_{i}" for i in range(1000)]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Advanced Image Processing Suite")
        self.setGeometry(100, 100, 1400, 900)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        self.create_menu_bar()
        self.create_toolbar()
        self.create_input_section()
        self.create_image_display()
        self.create_method_tabs()
        self.create_control_panel()
        self.create_status_bar()

        self.current_image = None
        self.processed_image = None
        self.video_capture = None
        self.processing_history = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def create_menu_bar(self):
        """Create the menu bar with File, Edit, View, and Help menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        actions = [
            ("Open Image", self.open_image),
            ("Open Video", self.open_video),
            ("Save Result", self.save_result),
            ("Export History", self.export_history),
            ("Exit", self.close)
        ]
        for text, slot in actions:
            action = file_menu.addAction(text)
            action.triggered.connect(slot)

        # Add other menus (Edit, View, Help) as needed

    def create_toolbar(self):
        """Create the toolbar with quick access buttons"""
        toolbar = self.addToolBar("Tools")
        toolbar.setMovable(False)

        # Add tool buttons
        self.add_toolbar_button(toolbar, "Open", self.open_image)
        self.add_toolbar_button(toolbar, "Save", self.save_result)
        self.add_toolbar_button(toolbar, "Reset", self.reset_processing)
        toolbar.addSeparator()
        self.add_toolbar_button(toolbar, "Zoom In", self.zoom_in)
        self.add_toolbar_button(toolbar, "Zoom Out", self.zoom_out)

    def create_input_section(self):
        """Create the input source selection section"""
        input_group = QGroupBox("Input Source")
        input_layout = QHBoxLayout()

        # Source selection combo
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Single Image", "Video File", "Webcam"])
        self.source_combo.currentTextChanged.connect(self.change_source)

        # Source selection button
        self.select_source_btn = QPushButton("Select Source")
        self.select_source_btn.clicked.connect(self.select_source)

        # Add widgets to layout
        input_layout.addWidget(QLabel("Source Type:"))
        input_layout.addWidget(self.source_combo)
        input_layout.addWidget(self.select_source_btn)
        input_layout.addStretch()

        # Create resolution control for webcam
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_combo.setVisible(False)
        input_layout.addWidget(QLabel("Resolution:"))
        input_layout.addWidget(self.resolution_combo)

        input_group.setLayout(input_layout)
        self.main_layout.addWidget(input_group)

    def create_image_display(self):
        """Create the image display area with input and output views"""
        display_layout = QHBoxLayout()

        # ===== INPUT IMAGE =====
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout()

        self.input_scroll = QScrollArea()
        self.input_image_label = QLabel()
        self.input_image_label.setFixedSize(400, 300)
        self.input_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_scroll.setWidget(self.input_image_label)
        self.input_scroll.setWidgetResizable(True)

        self.input_static_label = QLabel()
        self.input_static_label.setFixedSize(400, 300)
        self.input_static_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.input_stack = QStackedLayout()
        self.input_stack.addWidget(self.input_static_label)  # index 0
        self.input_stack.addWidget(self.input_scroll)  # index 1
        input_layout.addLayout(self.input_stack)
        input_group.setLayout(input_layout)

        # ===== OUTPUT IMAGE =====
        output_group = QGroupBox("Processed Image")
        output_layout = QVBoxLayout()

        self.output_scroll = QScrollArea()
        self.output_image_label = QLabel()
        self.output_image_label.setFixedSize(400, 300)
        self.output_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_scroll.setWidget(self.output_image_label)
        self.output_scroll.setWidgetResizable(True)

        self.output_static_label = QLabel()
        self.output_static_label.setFixedSize(400, 300)
        self.output_static_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.output_stack = QStackedLayout()
        self.output_stack.addWidget(self.output_static_label)  # index 0
        self.output_stack.addWidget(self.output_scroll)  # index 1
        output_layout.addLayout(self.output_stack)
        output_group.setLayout(output_layout)

        display_layout.addWidget(input_group)
        display_layout.addWidget(output_group)
        self.main_layout.addLayout(display_layout)

        # ===== GEOMETRIC INPUTS =====
        self.epipolar_group = QGroupBox("2 Input Page")
        epipolar_layout = QHBoxLayout()
        epipolar_layout.addWidget(self.epipolar_input_left_label)
        epipolar_layout.addWidget(self.epipolar_input_right_label)
        epipolar_layout.addWidget(self.epipolar_output_label)
        self.epipolar_group.setLayout(epipolar_layout)
        self.main_layout.addWidget(self.epipolar_group)
        self.epipolar_group.setVisible(False)

    def mouse_move_event(self, event):
        """Handle mouse movement over the input image and display pixel values in a specific order."""
        if self.current_image is not None:
            x = event.position().x()
            y = event.position().y()

            # QLabel içindeki koordinatı, orijinal görüntü boyutlarına ölçekleme
            label_width = self.input_image_label.width()
            label_height = self.input_image_label.height()
            img_height, img_width = self.current_image.shape[:2]

            if 0 <= x < label_width and 0 <= y < label_height:
                # Orijinal görüntüdeki koordinatları hesapla
                orig_x = int((x / label_width) * img_width)
                orig_y = int((y / label_height) * img_height)

                # Eğer grayscale görüntü ise
                if len(self.current_image.shape) == 2:
                    pixel_value = self.current_image[orig_y, orig_x]
                    pixel_info = f"X: {orig_x}, Y: {orig_y} | Grayscale: {pixel_value}"

                else:  # Renkli görüntü (BGR)
                    pixel_bgr = self.current_image[orig_y, orig_x]
                    b, g, r = pixel_bgr  # OpenCV varsayılan olarak BGR formatında
                    pixel_rgb = (r, g, b)  # RGB formatına çevir

                    # HSV formatına dönüştürme
                    pixel_bgr_reshaped = np.uint8([[pixel_bgr]])  # OpenCV için uygun formata getir
                    pixel_hsv = cv2.cvtColor(pixel_bgr_reshaped, cv2.COLOR_BGR2HSV)[0][0]
                    h, s, v = pixel_hsv

                    # Grayscale değeri hesapla
                    pixel_gray = cv2.cvtColor(pixel_bgr_reshaped, cv2.COLOR_BGR2GRAY)[0][0]

                    # **Updated Order: Pixel Position → RGB → HSV → BGR → Grayscale**
                    pixel_info = (
                        f"X: {orig_x}, Y: {orig_y} | "
                        f"RGB: ({r}, {g}, {b}) | "
                        f"HSV: ({h}, {s}, {v}) | "
                        f"BGR: ({b}, {g}, {r}) | "
                        f"Grayscale: {pixel_gray}"
                    )

                # Piksel bilgisini status bar’a yazdır
                self.status_bar.showMessage(pixel_info)

    def create_method_tabs(self):
        """Create tabs for different processing method categories"""
        self.tab_widget = QTabWidget()

        # Create tabs with scroll areas
        tabs_data = [
            ("Classical Methods", self.create_classical_tab),
            ("Geometric Methods", self.create_geometric_tab),
            ("Modern Methods", self.create_modern_tab),
            ("Deep Learning", self.create_deep_learning_tab)
        ]

        for tab_name, create_func in tabs_data:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.main_layout.addWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        tab_text = self.tab_widget.tabText(index)
        if tab_text == "Geometric Methods":
            self.input_stack.setCurrentIndex(1)  # scrollable QLabel
            self.output_stack.setCurrentIndex(1)
            self.epipolar_group.setVisible(True)
        else:
            self.input_stack.setCurrentIndex(0)  # static QLabel
            self.output_stack.setCurrentIndex(0)
            self.epipolar_group.setVisible(False)

    #KLASİK YÖNTEMLER
    def create_classical_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)

        filtering_group = QGroupBox("Filtering and Color Space")
        filtering_layout = QGridLayout()

        # Bold font tanımı
        bold_font = QLabel().font()
        bold_font.setBold(True)

        # Gaussian Kernel Slider
        self.gaussian_checkbox = QCheckBox("Gaussian Blur")
        self.gaussian_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_slider.setMinimum(1)
        self.gaussian_slider.setMaximum(15)
        self.gaussian_slider.setValue(7)
        self.gaussian_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gaussian_slider.setTickInterval(2)
        self.gaussian_slider.setFixedWidth(90)

        # Gaussian Sigma Slider
        self.gaussian_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_sigma_slider.setMinimum(0)
        self.gaussian_sigma_slider.setMaximum(250)
        self.gaussian_sigma_slider.setValue(0)
        self.gaussian_sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gaussian_sigma_slider.setTickInterval(10)
        self.gaussian_sigma_slider.setFixedWidth(90)

        self.gaussian_slider.valueChanged.connect(self.on_gaussian_slider_change)
        self.gaussian_sigma_slider.valueChanged.connect(self.on_gaussian_slider_change)

        # Median Kernel Slider
        self.median_checkbox = QCheckBox("Median Filter")
        self.median_slider = QSlider(Qt.Orientation.Horizontal)
        self.median_slider.setMinimum(1)
        self.median_slider.setMaximum(15)
        self.median_slider.setValue(7)
        self.median_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.median_slider.setTickInterval(2)
        self.median_slider.setFixedWidth(90)

        self.median_slider.valueChanged.connect(self.on_median_slider_change)

        # Bilateral Sigma Slider
        self.bilateral_checkbox = QCheckBox("Bilateral Filter")
        self.bilateral_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_slider.setMinimum(25)
        self.bilateral_slider.setMaximum(250)
        self.bilateral_slider.setValue(75)
        self.bilateral_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bilateral_slider.setTickInterval(10)
        self.bilateral_slider.setFixedWidth(90)

        # Bilateral Kernel Slider
        self.bilateral_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.bilateral_kernel_slider.setMinimum(1)
        self.bilateral_kernel_slider.setMaximum(15)
        self.bilateral_kernel_slider.setValue(9)
        self.bilateral_kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bilateral_kernel_slider.setTickInterval(2)
        self.bilateral_kernel_slider.setFixedWidth(90)

        self.bilateral_slider.valueChanged.connect(self.on_bilateral_slider_change)
        self.bilateral_kernel_slider.valueChanged.connect(self.on_bilateral_slider_change)

        # Grid layout for filters
        filter_controls_layout = QGridLayout()
        filter_controls_layout.setHorizontalSpacing(10)
        filter_controls_layout.setVerticalSpacing(5)

        # Gaussian row
        filter_controls_layout.addWidget(self.gaussian_checkbox, 0, 0)
        g_kernel_label = QLabel("Kernel:")
        g_kernel_label.setFont(bold_font)
        filter_controls_layout.addWidget(g_kernel_label, 0, 1)
        filter_controls_layout.addWidget(self.gaussian_slider, 0, 2)
        g_sigma_label = QLabel("Sigma:")
        g_sigma_label.setFont(bold_font)
        filter_controls_layout.addWidget(g_sigma_label, 0, 3)
        filter_controls_layout.addWidget(self.gaussian_sigma_slider, 0, 4)

        # Median row
        filter_controls_layout.addWidget(self.median_checkbox, 1, 0)
        m_kernel_label = QLabel("Kernel:")
        m_kernel_label.setFont(bold_font)
        filter_controls_layout.addWidget(m_kernel_label, 1, 1)
        filter_controls_layout.addWidget(self.median_slider, 1, 2)

        # Bilateral row
        filter_controls_layout.addWidget(self.bilateral_checkbox, 2, 0)
        b_kernel_label = QLabel("Kernel:")
        b_kernel_label.setFont(bold_font)
        filter_controls_layout.addWidget(b_kernel_label, 2, 1)
        filter_controls_layout.addWidget(self.bilateral_kernel_slider, 2, 2)
        b_sigma_label = QLabel("Sigma:")
        b_sigma_label.setFont(bold_font)
        filter_controls_layout.addWidget(b_sigma_label, 2, 3)
        filter_controls_layout.addWidget(self.bilateral_slider, 2, 4)

        # Bu layout'u ana filtering layout'a ekle
        filtering_layout.addLayout(filter_controls_layout, 0, 0, 1, 2)

        self.rgb_checkbox = QCheckBox("RGB")
        self.hsv_checkbox = QCheckBox("HSV")
        self.bgr_checkbox = QCheckBox("BGR")
        self.grayscale_checkbox = QCheckBox("Grayscale")

        # Gaussian Sigma değeri slider
        self.gaussian_sigma_slider.setMinimum(0)
        self.gaussian_sigma_slider.setMaximum(100)
        self.gaussian_sigma_slider.setValue(0)
        self.gaussian_sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gaussian_sigma_slider.setTickInterval(10)
        self.gaussian_sigma_slider.setFixedWidth(90)  # Daha kısa

        # Median Filter için slider ve bold Kernel etiketi
        self.median_slider.setMinimum(1)
        self.median_slider.setMaximum(15)
        self.median_slider.setValue(7)
        self.median_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.median_slider.setTickInterval(2)
        self.median_slider.setFixedWidth(90)  # Gaussian ile aynı

        # Bold font tanımı
        median_kernel_label = QLabel("Kernel:")
        bold_font = median_kernel_label.font()
        bold_font.setBold(True)
        median_kernel_label.setFont(bold_font)

        # Bilateral Filter için Sigma slider
        self.bilateral_slider.setMinimum(25)
        self.bilateral_slider.setMaximum(125)
        self.bilateral_slider.setValue(75)
        self.bilateral_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bilateral_slider.setTickInterval(10)
        self.bilateral_slider.setFixedWidth(90)

        # Renk uzayı checkbox’ları filtrelerle aynı satırın sağına hizalanacak şekilde ekleniyor
        filter_controls_layout.addWidget(self.rgb_checkbox, 0, 5, alignment=Qt.AlignmentFlag.AlignLeft)
        filter_controls_layout.addWidget(self.hsv_checkbox, 1, 5, alignment=Qt.AlignmentFlag.AlignLeft)
        filter_controls_layout.addWidget(self.bgr_checkbox, 2, 5, alignment=Qt.AlignmentFlag.AlignLeft)
        filter_controls_layout.addWidget(self.grayscale_checkbox, 3, 5, alignment=Qt.AlignmentFlag.AlignLeft)

        filtering_group.setLayout(filtering_layout)
        layout.addWidget(filtering_group, 0, 0)

        # Edge and Corner Detection Group
        edge_group = QGroupBox("Edge and Corner Detection")
        edge_layout = QGridLayout()

        self.sobel_checkbox = QCheckBox("Sobel")
        # Sobel yön seçimi kutusu
        self.sobel_direction_combo = QComboBox()
        self.sobel_direction_combo.addItems(["X", "Y", "Both"])
        self.sobel_direction_combo.currentTextChanged.connect(self.on_sobel_param_change)

        #Shi-Tomasi
        self.shi_tomasi_checkbox = QCheckBox("Shi-Tomasi")
        # Quality Level Slider for Shi-Tomasi
        quality_label = QLabel("QualityL:")
        quality_label.setFont(bold_font)
        self.shi_tomasi_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.shi_tomasi_quality_slider.valueChanged.connect(self.on_shi_tomasi_slider_change)
        self.shi_tomasi_quality_slider.setMinimum(1)
        self.shi_tomasi_quality_slider.setMaximum(100)
        self.shi_tomasi_quality_slider.setValue(10)  # Default = 0.10
        self.shi_tomasi_quality_slider.setFixedWidth(90)
        self.shi_tomasi_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.shi_tomasi_quality_slider.setTickInterval(10)
        edge_layout.addWidget(quality_label, 0, 9)
        edge_layout.addWidget(self.shi_tomasi_quality_slider, 0, 10)

        # FAST QualityL Slider (Threshold)
        fast_quality_label = QLabel("QualityL:")
        fast_quality_label.setFont(bold_font)
        self.fast_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.fast_quality_slider.setMinimum(1)
        self.fast_quality_slider.setMaximum(100)
        self.fast_quality_slider.setValue(10)
        self.fast_quality_slider.setFixedWidth(90)
        self.fast_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fast_quality_slider.setTickInterval(10)
        self.fast_quality_slider.valueChanged.connect(self.on_fast_slider_change)
        edge_layout.addWidget(fast_quality_label, 1, 9)
        edge_layout.addWidget(self.fast_quality_slider, 1, 10)

        #ORB Slider
        orb_quality_label = QLabel("QualityL:")
        orb_quality_label.setFont(bold_font)
        self.orb_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.orb_quality_slider.setMinimum(1)
        self.orb_quality_slider.setMaximum(100)
        self.orb_quality_slider.setValue(10)
        self.orb_quality_slider.setFixedWidth(90)
        self.orb_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.orb_quality_slider.setTickInterval(10)
        self.orb_quality_slider.valueChanged.connect(self.on_orb_slider_change)
        edge_layout.addWidget(orb_quality_label, 2, 9)
        edge_layout.addWidget(self.orb_quality_slider, 2, 10)

        #Canny
        self.canny_checkbox = QCheckBox("Canny")
        self.canny_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.canny_kernel_slider.setMinimum(0)
        self.canny_kernel_slider.setMaximum(2)  # 0 -> 3, 1 -> 5, 2 -> 7
        self.canny_kernel_slider.setTickInterval(1)
        self.canny_kernel_slider.setValue(0)
        self.canny_kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.canny_kernel_slider.setSingleStep(1)
        self.canny_kernel_slider.setFixedWidth(90)
        canny_kernel_label = QLabel("Kernel:")
        canny_kernel_label.setFont(bold_font)
        self.canny_kernel_slider.valueChanged.connect(self.on_canny_param_change)

        #Laplacian
        self.fast_checkbox = QCheckBox("FAST")
        self.laplacian_checkbox = QCheckBox("Laplacian")
        self.laplacian_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.laplacian_kernel_slider.setMinimum(1)
        self.laplacian_kernel_slider.setMaximum(5)
        self.laplacian_kernel_slider.setTickInterval(1)
        self.laplacian_kernel_slider.setValue(1)
        self.laplacian_kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.laplacian_kernel_slider.setSingleStep(1)
        self.laplacian_kernel_slider.setFixedWidth(90)
        laplacian_kernel_label = QLabel("Kernel:")
        laplacian_kernel_label.setFont(bold_font)
        self.laplacian_kernel_slider.valueChanged.connect(self.on_laplacian_param_change)

        self.orb_checkbox = QCheckBox("ORB")

        self.sobel_checkbox.setEnabled(True)
        self.shi_tomasi_checkbox.setEnabled(True)
        self.canny_checkbox.setEnabled(True)
        self.fast_checkbox.setEnabled(True)
        self.laplacian_checkbox.setEnabled(True)
        self.orb_checkbox.setEnabled(True)

        edge_layout.addWidget(self.sobel_checkbox, 0, 0)
        gdir_label = QLabel("GradientD:")
        bold_font = gdir_label.font()
        bold_font.setBold(True)
        gdir_label.setFont(bold_font)
        edge_layout.addWidget(gdir_label, 0, 1)
        edge_layout.addWidget(self.sobel_direction_combo, 0, 2)

        edge_layout.addWidget(self.shi_tomasi_checkbox, 0, 8)
        edge_layout.addWidget(self.canny_checkbox, 1, 0)
        edge_layout.addWidget(canny_kernel_label, 1, 1)
        edge_layout.addWidget(self.canny_kernel_slider, 1, 2)

        # Alt eşik (Lower Threshold) etiketi ve kutusu
        canny_lthres_label = QLabel("L.Thres(0/250):")
        canny_lthres_label.setFont(bold_font)
        self.canny_lthres_spinbox = QSpinBox()
        self.canny_lthres_spinbox.setRange(0, 255)
        self.canny_lthres_spinbox.setValue(50)
        self.canny_lthres_spinbox.setFixedWidth(60)
        self.canny_lthres_spinbox.valueChanged.connect(self.on_canny_param_change)

        # Üst eşik (Upper Threshold) etiketi ve kutusu
        canny_uthres_label = QLabel("U.Thres(0/250):")
        canny_uthres_label.setFont(bold_font)
        self.canny_uthres_spinbox = QSpinBox()
        self.canny_uthres_spinbox.setRange(0, 255)
        self.canny_uthres_spinbox.setValue(150)
        self.canny_uthres_spinbox.setFixedWidth(60)
        self.canny_uthres_spinbox.valueChanged.connect(self.on_canny_param_change)

        edge_layout.addWidget(canny_lthres_label, 1, 3)
        edge_layout.addWidget(self.canny_lthres_spinbox, 1, 4)
        edge_layout.addWidget(self.fast_checkbox, 1, 8)
        edge_layout.addWidget(canny_uthres_label, 1, 6)
        edge_layout.addWidget(self.canny_uthres_spinbox, 1, 7)
        edge_layout.addWidget(self.laplacian_checkbox, 2, 0)
        edge_layout.addWidget(laplacian_kernel_label, 2, 1)
        edge_layout.addWidget(self.laplacian_kernel_slider, 2, 2)
        edge_layout.addWidget(self.orb_checkbox, 2, 8)

        #Prewitt
        self.prewitt_checkbox = QCheckBox("Prewitt")
        self.prewitt_checkbox.setEnabled(True)
        edge_layout.addWidget(self.prewitt_checkbox, 3, 0)
        self.prewitt_direction_combo = QComboBox()
        self.prewitt_direction_combo.addItems(["X", "Y", "Both"])
        gdir_prewitt_label = QLabel("GradientD:")
        gdir_prewitt_label.setFont(bold_font)
        edge_layout.addWidget(gdir_prewitt_label, 3, 1)
        edge_layout.addWidget(self.prewitt_direction_combo, 3, 2)
        self.prewitt_direction_combo.currentTextChanged.connect(self.on_prewitt_param_change)

        edge_group.setLayout(edge_layout)
        layout.addWidget(edge_group, 0, 1)

        filtering_group.setMinimumWidth(edge_group.sizeHint().width())

        # Morphological Operations Group (Erosion, Dilation, Opening, Closing)
        morph_group = QGroupBox("Morphological")
        morph_layout = QGridLayout()

        #Erosion
        self.erosion_checkbox = QCheckBox("Erosion")
        self.erosion_checkbox.setEnabled(True)
        morph_layout.addWidget(self.erosion_checkbox, 0, 0)
        erosion_kernel_label = QLabel("Kernel:")
        erosion_kernel_label.setFont(bold_font)
        morph_layout.addWidget(erosion_kernel_label, 0, 1)
        self.erosion_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.erosion_kernel_slider.setMinimum(1)
        self.erosion_kernel_slider.setMaximum(15)
        self.erosion_kernel_slider.setValue(5)
        self.erosion_kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.erosion_kernel_slider.setTickInterval(2)
        self.erosion_kernel_slider.setFixedWidth(90)
        self.erosion_kernel_slider.valueChanged.connect(self.on_erosion_slider_change)
        morph_layout.addWidget(self.erosion_kernel_slider, 0, 2)

        #Dilation
        self.dilation_checkbox = QCheckBox("Dilation")
        self.dilation_checkbox.setEnabled(True)
        morph_layout.addWidget(self.dilation_checkbox, 1, 0)
        # Dilation Slider
        dilation_kernel_label = QLabel("Kernel:")
        dilation_kernel_label.setFont(bold_font)
        morph_layout.addWidget(dilation_kernel_label, 1, 1)
        self.dilation_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.dilation_kernel_slider.setMinimum(1)
        self.dilation_kernel_slider.setMaximum(15)
        self.dilation_kernel_slider.setValue(5)
        self.dilation_kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.dilation_kernel_slider.setTickInterval(2)
        self.dilation_kernel_slider.setFixedWidth(90)
        self.dilation_kernel_slider.valueChanged.connect(self.on_dilation_slider_change)
        morph_layout.addWidget(self.dilation_kernel_slider, 1, 2)

        #Opening
        self.opening_checkbox = QCheckBox("Opening")
        self.opening_checkbox.setEnabled(True)
        morph_layout.addWidget(self.opening_checkbox, 2, 0)
        # Opening Kernel Slider
        opening_kernel_label = QLabel("Kernel:")
        opening_kernel_label.setFont(bold_font)
        self.opening_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.opening_kernel_slider.setMinimum(1)
        self.opening_kernel_slider.setMaximum(15)
        self.opening_kernel_slider.setValue(5)
        self.opening_kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.opening_kernel_slider.setTickInterval(2)
        self.opening_kernel_slider.setFixedWidth(90)
        self.opening_kernel_slider.valueChanged.connect(self.on_opening_slider_change)
        morph_layout.addWidget(opening_kernel_label)
        morph_layout.addWidget(self.opening_kernel_slider)

        #Closing
        self.closing_checkbox = QCheckBox("Closing")
        self.closing_checkbox.setEnabled(True)
        morph_layout.addWidget(self.closing_checkbox, 3, 0)
        # Closing Kernel Slider
        closing_kernel_label = QLabel("Kernel:")
        closing_kernel_label.setFont(bold_font)
        self.closing_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.closing_kernel_slider.setMinimum(1)
        self.closing_kernel_slider.setMaximum(15)
        self.closing_kernel_slider.setValue(5)
        self.closing_kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.closing_kernel_slider.setTickInterval(2)
        self.closing_kernel_slider.setFixedWidth(90)
        self.closing_kernel_slider.valueChanged.connect(self.on_closing_slider_change)
        morph_layout.addWidget(closing_kernel_label)
        morph_layout.addWidget(self.closing_kernel_slider)

        morph_group.setLayout(morph_layout)

        layout.addWidget(morph_group, 1, 0)

        # Line and Circle Detection Group (New Box)
        line_circle_group = QGroupBox("Line and Circle Detection")
        line_circle_layout = QGridLayout()

        self.hough_lines_checkbox = QCheckBox("Hough Lines")
        self.hough_circles_checkbox = QCheckBox("Hough Circles")

        self.hough_lines_checkbox.setEnabled(True)
        self.hough_circles_checkbox.setEnabled(True)

        line_circle_layout.addWidget(self.hough_lines_checkbox, 0, 0)
        line_circle_layout.addWidget(self.hough_circles_checkbox, 1, 0)

        line_circle_group.setLayout(line_circle_layout)
        layout.addWidget(line_circle_group, 1, 1)

        # Hough Lines Rho Slider
        rho_label = QLabel("Rho:")
        bold_font = rho_label.font()
        bold_font.setBold(True)
        rho_label.setFont(bold_font)
        self.hough_rho_slider = QSlider(Qt.Orientation.Horizontal)
        self.hough_rho_slider.setMinimum(1)
        self.hough_rho_slider.setMaximum(5)
        self.hough_rho_slider.setValue(1)
        self.hough_rho_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.hough_rho_slider.setTickInterval(1)
        self.hough_rho_slider.setFixedWidth(90)
        self.hough_rho_slider.valueChanged.connect(self.on_hough_slider_change)
        line_circle_layout.addWidget(rho_label, 0, 1)  # Rho: label next to Hough Lines
        line_circle_layout.addWidget(self.hough_rho_slider, 0, 2)  # Rho slider next to label

        # Hough Lines Theta Slider
        theta_label = QLabel("Theta:")
        bold_font = theta_label.font()
        bold_font.setBold(True)
        theta_label.setFont(bold_font)
        line_circle_layout.addWidget(theta_label, 0, 4)
        self.hough_theta_slider = QSlider(Qt.Orientation.Horizontal)
        self.hough_theta_slider.setMinimum(1)  # 1 derece
        self.hough_theta_slider.setMaximum(180)  # 180 derece
        self.hough_theta_slider.setValue(1)
        self.hough_theta_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.hough_theta_slider.setTickInterval(10)
        self.hough_theta_slider.setFixedWidth(90)
        self.hough_theta_slider.setSingleStep(1)
        self.hough_theta_slider.valueChanged.connect(self.on_hough_slider_change)
        line_circle_layout.addWidget(self.hough_theta_slider, 0, 5)

        # Hough Lines Threshold Slider
        threshold_label = QLabel("Threshold:")
        threshold_label.setFont(bold_font)
        self.hough_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.hough_threshold_slider.setMinimum(50)
        self.hough_threshold_slider.setMaximum(300)
        self.hough_threshold_slider.setValue(150)
        self.hough_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.hough_threshold_slider.setTickInterval(10)
        self.hough_threshold_slider.setFixedWidth(90)
        self.hough_threshold_slider.valueChanged.connect(self.on_hough_slider_change)
        line_circle_layout.addWidget(threshold_label, 0, 6)
        line_circle_layout.addWidget(self.hough_threshold_slider, 0, 7)

        # --- HOUGH CIRCLES PARAMETRELERİ ---

        # Param1: Canny Upper Threshold
        param1_label = QLabel("CannyThresh:")
        param1_label.setFont(bold_font)
        self.hough_param1_slider = QSlider(Qt.Orientation.Horizontal)
        self.hough_param1_slider.setMinimum(10)
        self.hough_param1_slider.setMaximum(300)
        self.hough_param1_slider.setValue(50)
        self.hough_param1_slider.setTickInterval(10)
        self.hough_param1_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.hough_param1_slider.setFixedWidth(90)
        self.hough_param1_slider.valueChanged.connect(self.on_hough_slider_change)
        line_circle_layout.addWidget(param1_label, 1, 1)
        line_circle_layout.addWidget(self.hough_param1_slider, 1, 2)

        # Param2: Circle Center Vote Threshold
        param2_label = QLabel("VoteThresh:")
        param2_label.setFont(bold_font)
        self.hough_param2_slider = QSlider(Qt.Orientation.Horizontal)
        self.hough_param2_slider.setMinimum(10)
        self.hough_param2_slider.setMaximum(100)
        self.hough_param2_slider.setValue(30)
        self.hough_param2_slider.setTickInterval(10)
        self.hough_param2_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.hough_param2_slider.setFixedWidth(90)
        self.hough_param2_slider.valueChanged.connect(self.on_hough_slider_change)
        line_circle_layout.addWidget(param2_label, 1, 4)
        line_circle_layout.addWidget(self.hough_param2_slider, 1, 5)

        # minRadius
        min_radius_label = QLabel("minR(0/200):")
        min_radius_label.setFont(bold_font)
        self.hough_min_radius_spinbox = QSpinBox()
        self.hough_min_radius_spinbox.setRange(0, 200)
        self.hough_min_radius_spinbox.setValue(10)
        self.hough_min_radius_spinbox.setFixedWidth(60)
        self.hough_min_radius_spinbox.valueChanged.connect(self.on_hough_slider_change)
        line_circle_layout.addWidget(min_radius_label, 1, 6)
        line_circle_layout.addWidget(self.hough_min_radius_spinbox, 1, 7)

        # maxRadius
        max_radius_label = QLabel("maxR(0/300):")
        max_radius_label.setFont(bold_font)
        self.hough_max_radius_spinbox = QSpinBox()
        self.hough_max_radius_spinbox.setRange(0, 300)
        self.hough_max_radius_spinbox.setValue(100)
        self.hough_max_radius_spinbox.setFixedWidth(60)
        self.hough_max_radius_spinbox.valueChanged.connect(self.on_hough_slider_change)
        line_circle_layout.addWidget(max_radius_label, 1, 8)
        line_circle_layout.addWidget(self.hough_max_radius_spinbox, 1, 9)

        return widget

    def on_gaussian_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_median_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_bilateral_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_sobel_param_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_canny_param_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_laplacian_param_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_prewitt_param_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_shi_tomasi_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_fast_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_orb_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_erosion_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_dilation_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_opening_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_closing_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_hough_slider_change(self):
        if self.is_image_loaded():
            self.process_image()


    #GEOMETRİK YÖNTEMLER
    def create_geometric_tab(self):
        """Create the geometric methods tab content"""
        widget = QWidget()
        layout = QGridLayout(widget)
        bold_font = QLabel().font()
        bold_font.setBold(True)

        groups = [
            ("Basic Transforms", ["Resize", "Rotate", "Flip"]),
            ("Advanced Transforms", ["Affine", "Perspective", "Warp"]),
            ("Features", ["Corner Detection", "Line Detection", "Contours"])
        ]
        groups.append(("Advanced Functions", ["Manual Homography"]))

        for i, (group_name, methods) in enumerate(groups):
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()

            for method in methods:
                #Resize
                if method == "Resize":
                    self.resize_checkbox = QCheckBox("Resize")
                    resize_row_layout = QHBoxLayout()
                    resize_row_layout.addWidget(self.resize_checkbox)

                    resize_row_layout.addSpacing(10)
                    width_label = QLabel("Width:")
                    width_label.setFont(bold_font)
                    resize_row_layout.addWidget(width_label)
                    self.resize_width_box = QSpinBox()
                    self.resize_width_box.setRange(1, 5000)
                    self.resize_width_box.setValue(512)
                    resize_row_layout.addWidget(self.resize_width_box)

                    resize_row_layout.addSpacing(10)
                    height_label = QLabel("Height:")
                    height_label.setFont(bold_font)
                    resize_row_layout.addWidget(height_label)
                    self.resize_height_box = QSpinBox()
                    self.resize_height_box.setRange(1, 5000)
                    self.resize_height_box.setValue(512)
                    resize_row_layout.addWidget(self.resize_height_box)

                    group_layout.addLayout(resize_row_layout)

                    # 1. QLabel ve QComboBox tanımları
                    interpolation_label = QLabel("Interpolation:")
                    bold_font = interpolation_label.font()
                    bold_font.setBold(True)
                    interpolation_label.setFont(bold_font)

                    self.interpolation_combo = QComboBox()
                    self.interpolation_combo.addItems(["None", "Nearest-neighbor", "Bilinear", "Bicubic"])

                    # 2. Resize satırına ekleme (resize_row_layout)
                    resize_row_layout.addSpacing(10)
                    resize_row_layout.addWidget(interpolation_label)
                    resize_row_layout.addWidget(self.interpolation_combo)

                #Rotate
                elif method == "Rotate":
                    self.rotate_checkbox = QCheckBox("Rotate")
                    rotate_row_layout = QHBoxLayout()
                    rotate_row_layout.addWidget(self.rotate_checkbox)

                    rotate_row_layout.addSpacing(10)
                    angle_label = QLabel("Angle:")
                    angle_label.setFont(bold_font)
                    rotate_row_layout.addWidget(angle_label)
                    self.rotate_angle_spinbox = QSpinBox()
                    self.rotate_angle_spinbox.setRange(-360, 360)
                    self.rotate_angle_spinbox.setValue(0)
                    rotate_row_layout.addWidget(self.rotate_angle_spinbox)

                    rotate_row_layout.addSpacing(10)
                    pivotx_label = QLabel("Pivot X:")
                    pivotx_label.setFont(bold_font)
                    rotate_row_layout.addWidget(pivotx_label)
                    self.rotate_pivot_x = QSpinBox()
                    self.rotate_pivot_x.setRange(0, 5000)
                    self.rotate_pivot_x.setValue(0)
                    rotate_row_layout.addWidget(self.rotate_pivot_x)

                    rotate_row_layout.addSpacing(10)
                    pivoty_label = QLabel("Pivot Y:")
                    pivoty_label.setFont(bold_font)
                    rotate_row_layout.addWidget(pivoty_label)
                    self.rotate_pivot_y = QSpinBox()
                    self.rotate_pivot_y.setRange(0, 5000)
                    self.rotate_pivot_y.setValue(0)
                    rotate_row_layout.addWidget(self.rotate_pivot_y)

                    group_layout.addLayout(rotate_row_layout)

                #Flip
                elif method == "Flip":
                    self.flip_checkbox = QCheckBox("Flip (Reflection)")
                    flip_row_layout = QHBoxLayout()
                    flip_row_layout.addWidget(self.flip_checkbox)

                    flip_row_layout.addSpacing(10)
                    axis_label = QLabel("Axis:")
                    axis_label.setFont(bold_font)
                    flip_row_layout.addWidget(axis_label)
                    self.flip_axis_combo = QComboBox()
                    self.flip_axis_combo.addItems(["X", "Y", "Both"])
                    flip_row_layout.addWidget(self.flip_axis_combo)

                    group_layout.addLayout(flip_row_layout)

                    #Translate
                    self.translation_checkbox = QCheckBox("Translation")
                    translation_row_layout = QHBoxLayout()
                    translation_row_layout.addWidget(self.translation_checkbox)

                    translation_row_layout.addSpacing(10)
                    xoffset_label = QLabel("X Offset:")
                    xoffset_label.setFont(bold_font)
                    translation_row_layout.addWidget(xoffset_label)
                    self.translation_x_spinbox = QSpinBox()
                    self.translation_x_spinbox.setRange(-1000, 1000)
                    self.translation_x_spinbox.setValue(0)
                    translation_row_layout.addWidget(self.translation_x_spinbox)

                    translation_row_layout.addSpacing(10)
                    yoffset_label = QLabel("Y Offset:")
                    yoffset_label.setFont(bold_font)
                    translation_row_layout.addWidget(yoffset_label)
                    self.translation_y_spinbox = QSpinBox()
                    self.translation_y_spinbox.setRange(-1000, 1000)
                    self.translation_y_spinbox.setValue(0)
                    translation_row_layout.addWidget(self.translation_y_spinbox)

                    group_layout.addLayout(translation_row_layout)

                    #Scale
                    self.scaling_checkbox = QCheckBox("Scaling")
                    scaling_row_layout = QHBoxLayout()
                    scaling_row_layout.addWidget(self.scaling_checkbox)

                    scaling_row_layout.addSpacing(10)
                    scalex_label = QLabel("Scale X:")
                    scalex_label.setFont(bold_font)
                    scaling_row_layout.addWidget(scalex_label)
                    self.scale_x_spinbox = QDoubleSpinBox()
                    self.scale_x_spinbox.setRange(0.1, 5.0)
                    self.scale_x_spinbox.setSingleStep(0.1)
                    self.scale_x_spinbox.setValue(1.0)
                    scaling_row_layout.addWidget(self.scale_x_spinbox)

                    scaling_row_layout.addSpacing(10)
                    scaley_label = QLabel("Scale Y:")
                    scaley_label.setFont(bold_font)
                    scaling_row_layout.addWidget(scaley_label)
                    self.scale_y_spinbox = QDoubleSpinBox()
                    self.scale_y_spinbox.setRange(0.1, 5.0)
                    self.scale_y_spinbox.setSingleStep(0.1)
                    self.scale_y_spinbox.setValue(1.0)
                    scaling_row_layout.addWidget(self.scale_y_spinbox)

                    self.uniform_scaling_checkbox = QCheckBox("Uniform Scaling")
                    self.uniform_scaling_checkbox.setFont(bold_font)
                    self.uniform_scaling_checkbox.setChecked(True)
                    self.uniform_scaling_checkbox.stateChanged.connect(self.sync_scale_y)
                    scaling_row_layout.addWidget(self.uniform_scaling_checkbox)

                    group_layout.addLayout(scaling_row_layout)

                    #Shear
                    self.shearing_checkbox = QCheckBox("Shearing")
                    shearing_row_layout = QHBoxLayout()
                    shearing_row_layout.addWidget(self.shearing_checkbox)

                    shearing_row_layout.addSpacing(10)
                    shearx_label = QLabel("Shear X:")
                    shearx_label.setFont(bold_font)
                    shearing_row_layout.addWidget(shearx_label)
                    self.shear_x_spinbox = QDoubleSpinBox()
                    self.shear_x_spinbox.setRange(-2.0, 2.0)
                    self.shear_x_spinbox.setSingleStep(0.1)
                    self.shear_x_spinbox.setValue(0.0)
                    shearing_row_layout.addWidget(self.shear_x_spinbox)

                    shearing_row_layout.addSpacing(10)
                    sheary_label = QLabel("Shear Y:")
                    sheary_label.setFont(bold_font)
                    shearing_row_layout.addWidget(sheary_label)
                    self.shear_y_spinbox = QDoubleSpinBox()
                    self.shear_y_spinbox.setRange(-2.0, 2.0)
                    self.shear_y_spinbox.setSingleStep(0.1)
                    self.shear_y_spinbox.setValue(0.0)
                    shearing_row_layout.addWidget(self.shear_y_spinbox)

                    group_layout.addLayout(shearing_row_layout)

                #Affine
                elif method == "Affine":
                    self.affine_checkbox = QCheckBox("Affine")
                    affine_row_layout = QHBoxLayout()
                    affine_row_layout.addWidget(self.affine_checkbox)

                    affine_row_layout.addSpacing(10)
                    affine_angle_label = QLabel("Angle:")
                    affine_angle_label.setFont(bold_font)
                    affine_row_layout.addWidget(affine_angle_label)
                    self.affine_angle_spinbox = QSpinBox()
                    self.affine_angle_spinbox.setRange(-360, 360)
                    self.affine_angle_spinbox.setValue(0)
                    affine_row_layout.addWidget(self.affine_angle_spinbox)

                    affine_row_layout.addSpacing(10)
                    affine_tx_label = QLabel("X Offset:")
                    affine_tx_label.setFont(bold_font)
                    affine_row_layout.addWidget(affine_tx_label)
                    self.affine_translation_x = QSpinBox()
                    self.affine_translation_x.setRange(-1000, 1000)
                    self.affine_translation_x.setValue(0)
                    affine_row_layout.addWidget(self.affine_translation_x)

                    affine_row_layout.addSpacing(10)
                    affine_ty_label = QLabel("Y Offset:")
                    affine_ty_label.setFont(bold_font)
                    affine_row_layout.addWidget(affine_ty_label)
                    self.affine_translation_y = QSpinBox()
                    self.affine_translation_y.setRange(-1000, 1000)
                    self.affine_translation_y.setValue(0)
                    affine_row_layout.addWidget(self.affine_translation_y)

                    affine_row_layout.addSpacing(10)
                    affine_scale_label = QLabel("Uniform Scale:")
                    affine_scale_label.setFont(bold_font)
                    affine_row_layout.addWidget(affine_scale_label)
                    self.affine_scale_spinbox = QDoubleSpinBox()
                    self.affine_scale_spinbox.setRange(0.1, 5.0)
                    self.affine_scale_spinbox.setSingleStep(0.1)
                    self.affine_scale_spinbox.setValue(1.0)
                    affine_row_layout.addWidget(self.affine_scale_spinbox)

                    group_layout.addLayout(affine_row_layout)

                    # 6DoF Affine
                    self.affine6dof_checkbox = QCheckBox("6DoF Affine")
                    affine6dof_row_layout = QHBoxLayout()
                    affine6dof_row_layout.addWidget(self.affine6dof_checkbox)

                    # 6 parametre (a11, a12, a13, a21, a22, a23) spinbox'ları
                    bold_font = QLabel().font()
                    bold_font.setBold(True)

                    # a11
                    a11_label = QLabel("a11:")
                    a11_label.setFont(bold_font)
                    affine6dof_row_layout.addWidget(a11_label)
                    self.affine6dof_a11_spinbox = QDoubleSpinBox()
                    self.affine6dof_a11_spinbox.setRange(-10.0, 10.0)
                    self.affine6dof_a11_spinbox.setDecimals(3)
                    self.affine6dof_a11_spinbox.setSingleStep(0.01)
                    self.affine6dof_a11_spinbox.setValue(1.0)
                    affine6dof_row_layout.addWidget(self.affine6dof_a11_spinbox)

                    # a12
                    a12_label = QLabel("a12:")
                    a12_label.setFont(bold_font)
                    affine6dof_row_layout.addWidget(a12_label)
                    self.affine6dof_a12_spinbox = QDoubleSpinBox()
                    self.affine6dof_a12_spinbox.setRange(-10.0, 10.0)
                    self.affine6dof_a12_spinbox.setDecimals(3)
                    self.affine6dof_a12_spinbox.setSingleStep(0.01)
                    self.affine6dof_a12_spinbox.setValue(0.0)
                    affine6dof_row_layout.addWidget(self.affine6dof_a12_spinbox)

                    # a13 (translation x)
                    a13_label = QLabel("a13:")
                    a13_label.setFont(bold_font)
                    affine6dof_row_layout.addWidget(a13_label)
                    self.affine6dof_a13_spinbox = QDoubleSpinBox()
                    self.affine6dof_a13_spinbox.setRange(-1000.0, 1000.0)
                    self.affine6dof_a13_spinbox.setDecimals(2)
                    self.affine6dof_a13_spinbox.setSingleStep(1.0)
                    self.affine6dof_a13_spinbox.setValue(0.0)
                    affine6dof_row_layout.addWidget(self.affine6dof_a13_spinbox)

                    # a21
                    a21_label = QLabel("a21:")
                    a21_label.setFont(bold_font)
                    affine6dof_row_layout.addWidget(a21_label)
                    self.affine6dof_a21_spinbox = QDoubleSpinBox()
                    self.affine6dof_a21_spinbox.setRange(-10.0, 10.0)
                    self.affine6dof_a21_spinbox.setDecimals(3)
                    self.affine6dof_a21_spinbox.setSingleStep(0.01)
                    self.affine6dof_a21_spinbox.setValue(0.0)
                    affine6dof_row_layout.addWidget(self.affine6dof_a21_spinbox)

                    # a22
                    a22_label = QLabel("a22:")
                    a22_label.setFont(bold_font)
                    affine6dof_row_layout.addWidget(a22_label)
                    self.affine6dof_a22_spinbox = QDoubleSpinBox()
                    self.affine6dof_a22_spinbox.setRange(-10.0, 10.0)
                    self.affine6dof_a22_spinbox.setDecimals(3)
                    self.affine6dof_a22_spinbox.setSingleStep(0.01)
                    self.affine6dof_a22_spinbox.setValue(1.0)
                    affine6dof_row_layout.addWidget(self.affine6dof_a22_spinbox)

                    # a23 (translation y)
                    a23_label = QLabel("a23:")
                    a23_label.setFont(bold_font)
                    affine6dof_row_layout.addWidget(a23_label)
                    self.affine6dof_a23_spinbox = QDoubleSpinBox()
                    self.affine6dof_a23_spinbox.setRange(-1000.0, 1000.0)
                    self.affine6dof_a23_spinbox.setDecimals(2)
                    self.affine6dof_a23_spinbox.setSingleStep(1.0)
                    self.affine6dof_a23_spinbox.setValue(0.0)
                    affine6dof_row_layout.addWidget(self.affine6dof_a23_spinbox)

                    group_layout.addLayout(affine6dof_row_layout)



                elif method == "Perspective":
                    perspective_row_layout = QHBoxLayout()
                    self.perspective_checkbox = QCheckBox("Perspective")
                    self.perspective_checkbox.stateChanged.connect(self.on_perspective_checkbox_change)
                    perspective_row_layout.addWidget(self.perspective_checkbox)
                    self.perspective_points_inputs = []
                    for i in range(4):
                        point_input = QLineEdit()
                        point_input.setPlaceholderText(f"x{i + 1},y{i + 1}")
                        point_input.setFixedWidth(80)
                        self.perspective_points_inputs.append(point_input)
                        perspective_row_layout.addWidget(point_input)
                    group_layout.addLayout(perspective_row_layout)


                #Corner Detection
                elif method == "Warp":
                    self.warp_checkbox = QCheckBox("Warp")
                    warp_row_layout = QHBoxLayout()
                    warp_row_layout.setSpacing(6)
                    warp_row_layout.addWidget(self.warp_checkbox)
                    warp_labels = ["w11:", "w12:", "w13:",
                                   "w21:", "w22:", "w23:",
                                   "w31:", "w32:", "w33:"]
                    self.warp_matrix_spinboxes = []
                    for i, label_text in enumerate(warp_labels):
                        label = QLabel(label_text)
                        bold_font = label.font()
                        bold_font.setBold(True)
                        label.setFont(bold_font)
                        warp_row_layout.addWidget(label)
                        box = QDoubleSpinBox()
                        box.setRange(-1000.0, 1000.0)
                        box.setDecimals(3)
                        box.setSingleStep(0.1)
                        box.setValue(1.0 if i in [0, 4, 8] else 0.0)  # Identity matrix varsayılanı
                        box.setFixedWidth(60)
                        self.warp_matrix_spinboxes.append(box)
                        warp_row_layout.addWidget(box)
                    warp_container = QWidget()
                    warp_container.setLayout(warp_row_layout)
                    group_layout.addWidget(warp_container)

                #Corner Detection
                elif method == "Corner Detection":
                    corner_row_layout = QHBoxLayout()
                    self.corner_detection_checkbox = QCheckBox("Corner Detection")
                    self.corner_detection_checkbox.setEnabled(True)
                    corner_row_layout.addWidget(self.corner_detection_checkbox)
                    corner_row_layout.addSpacing(10)
                    corner_quality_label = QLabel("QualityL:")
                    bold_font = corner_quality_label.font()
                    bold_font.setBold(True)
                    corner_quality_label.setFont(bold_font)
                    corner_row_layout.addWidget(corner_quality_label)
                    self.corner_quality_slider = QSlider(Qt.Orientation.Horizontal)
                    self.corner_quality_slider.setMinimum(1)
                    self.corner_quality_slider.setMaximum(100)
                    self.corner_quality_slider.setValue(10)
                    self.corner_quality_slider.setTickInterval(10)
                    self.corner_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.corner_quality_slider.setFixedWidth(90)
                    self.corner_quality_slider.valueChanged.connect(self.on_corner_slider_change)
                    corner_row_layout.addWidget(self.corner_quality_slider)
                    group_layout.addLayout(corner_row_layout)

                #Line Detection
                elif method == "Line Detection":
                    self.line_detection_checkbox = QCheckBox("Line Detection")
                    line_row_layout = QHBoxLayout()
                    line_row_layout.addWidget(self.line_detection_checkbox)
                    threshold_label = QLabel("Threshold:")
                    threshold_label.setFont(bold_font)
                    line_row_layout.addWidget(threshold_label)
                    self.line_threshold_slider = QSlider(Qt.Orientation.Horizontal)
                    self.line_threshold_slider.setMinimum(10)
                    self.line_threshold_slider.setMaximum(300)
                    self.line_threshold_slider.setValue(100)
                    self.line_threshold_slider.setTickInterval(10)
                    self.line_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.line_threshold_slider.setFixedWidth(90)
                    self.line_threshold_slider.valueChanged.connect(self.on_line_slider_change)
                    line_row_layout.addWidget(self.line_threshold_slider)
                    group_layout.addLayout(line_row_layout)

                #Contour
                elif method == "Contours":
                    self.contour_checkbox = QCheckBox("Contours")
                    contour_row_layout = QHBoxLayout()
                    contour_row_layout.addWidget(self.contour_checkbox)

                    contour_area_label = QLabel("Min Area:")
                    bold_font = contour_area_label.font()
                    bold_font.setBold(True)
                    contour_area_label.setFont(bold_font)

                    self.contour_area_slider = QSlider(Qt.Orientation.Horizontal)
                    self.contour_area_slider.setMinimum(10)
                    self.contour_area_slider.setMaximum(1000)
                    self.contour_area_slider.setValue(100)
                    self.contour_area_slider.setTickInterval(50)
                    self.contour_area_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.contour_area_slider.setFixedWidth(120)
                    self.contour_area_slider.valueChanged.connect(self.on_contour_slider_change)
                    contour_row_layout.addWidget(contour_area_label)
                    contour_row_layout.addWidget(self.contour_area_slider)
                    group_layout.addLayout(contour_row_layout)

                #Manuel Homography
                elif method == "Manual Homography":
                    self.manual_homography_checkbox = QCheckBox("Manual Homography")
                    homography_row_layout = QHBoxLayout()
                    homography_row_layout.addWidget(self.manual_homography_checkbox)
                    bold_font = QLabel().font()
                    bold_font.setBold(True)
                    self.homography_src_inputs = []
                    self.homography_dst_inputs = []

                    for i in range(4):
                        s_label = QLabel(f"source{i + 1}:")
                        s_label.setFont(bold_font)
                        s_input = QLineEdit()
                        s_input.setPlaceholderText("x,y")
                        s_input.setFixedWidth(60)
                        self.homography_src_inputs.append(s_input)
                        t_label = QLabel(f"target{i + 1}:")
                        t_label.setFont(bold_font)
                        t_input = QLineEdit()
                        t_input.setPlaceholderText("x,y")
                        t_input.setFixedWidth(60)
                        self.homography_dst_inputs.append(t_input)
                        homography_row_layout.addWidget(s_label)
                        homography_row_layout.addWidget(s_input)
                        homography_row_layout.addWidget(t_label)
                        homography_row_layout.addWidget(t_input)
                    group_layout.addLayout(homography_row_layout)
                    # Homography matrisi göstermek için QLabel
                    self.homography_matrix_label = QLabel("")
                    group_layout.addWidget(self.homography_matrix_label)

                    threshold_label = QLabel("RANSAC Threshold:")
                    threshold_label.setFont(bold_font)
                    self.ransac_threshold_spinbox = QDoubleSpinBox()
                    self.ransac_threshold_spinbox.setRange(0.1, 10.0)
                    self.ransac_threshold_spinbox.setSingleStep(0.1)
                    self.ransac_threshold_spinbox.setValue(3.0)  # default
                    self.ransac_threshold_spinbox.setFixedWidth(70)
                    homography_row_layout.addWidget(threshold_label)
                    homography_row_layout.addWidget(self.ransac_threshold_spinbox)

                    # Epipolar Geometry Estimation
                    epipolar_row = QHBoxLayout()
                    self.epipolar_checkbox = QCheckBox("Estimate Epipolar Geometry")
                    epipolar_row.addWidget(self.epipolar_checkbox)

                    self.epipolar_left_btn = QPushButton("Load Left Image")
                    self.epipolar_right_btn = QPushButton("Load Right Image")
                    self.plot_epipolar_lines_checkbox = QCheckBox("Plot Lines")

                    epipolar_row.addWidget(self.epipolar_left_btn)
                    epipolar_row.addWidget(self.epipolar_right_btn)
                    epipolar_row.addWidget(self.plot_epipolar_lines_checkbox)

                    self.epipolar_left_btn.clicked.connect(self.load_epipolar_left_image)
                    self.epipolar_right_btn.clicked.connect(self.load_epipolar_right_image)

                    group_layout.addLayout(epipolar_row)

                    combine_row = QHBoxLayout()
                    self.combine_homography_checkbox = QCheckBox("Combine using Homography")

                    self.combine_left_btn = QPushButton("Load Left Image")
                    self.combine_right_btn = QPushButton("Load Right Image")

                    self.combine_left_btn.clicked.connect(self.load_combine_left_image)
                    self.combine_right_btn.clicked.connect(self.load_combine_right_image)

                    combine_row.addWidget(self.combine_homography_checkbox)
                    combine_row.addWidget(self.combine_left_btn)
                    combine_row.addWidget(self.combine_right_btn)

                    group_layout.addLayout(combine_row)


                else:
                    checkbox = QCheckBox(method)
                    checkbox.setEnabled(False)
                    group_layout.addWidget(checkbox)

            group.setLayout(group_layout)
            layout.addWidget(group, i // 2, i % 2)

        return widget

    def load_combine_left_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Left Image for Combination", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.combine_left_image = cv2.imread(file_name)
            self.status_bar.showMessage("Left image loaded for homography combination.")

    def load_combine_right_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Right Image for Combination", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.combine_right_image = cv2.imread(file_name)
            self.status_bar.showMessage("Right image loaded for homography combination.")

    def load_epipolar_left_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Left Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            image = cv2.imread(file_name)
            if image is None:
                self.show_error("Left image could not be loaded. Please check the file format or path.")
                return
            self.epipolar_left_image = image
            self.status_bar.showMessage("Left image loaded for epipolar estimation.")
            self.display_image(self.epipolar_left_image, self.epipolar_input_left_label)

    def load_epipolar_right_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Right Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            image = cv2.imread(file_name)
            if image is None:
                self.show_error("Right image could not be loaded. Please check the file format or path.")
                return
            self.epipolar_right_image = image
            self.status_bar.showMessage("Right image loaded for epipolar estimation.")
            self.display_image(self.epipolar_right_image, self.epipolar_input_right_label)

    def on_contour_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_contour_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def on_line_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def reset_perspective_points(self):
        """Reset selected perspective points."""
        self.perspective_src_points.clear()
        if self.current_image is not None:
            self.display_image(self.current_image, self.input_image_label)
        self.status_bar.showMessage("Perspective points reset.")

    def on_perspective_checkbox_change(self):
        if self.perspective_checkbox.isChecked():
            self.status_bar.showMessage("Enter 4 (x,y) coordinates in the input boxes.")
            # Kullanıcının girdiği kutulardan koordinatları oku
            self.perspective_src_points.clear()
            for box in self.perspective_points_inputs:
                try:
                    x_str, y_str = box.text().split(',')
                    self.perspective_src_points.append([float(x_str), float(y_str)])
                except ValueError:
                    self.show_error("Invalid point format. Use x,y format like 100,200.")
                    self.perspective_checkbox.setChecked(False)
                    return

    def sync_scale_y(self):
        if self.uniform_scaling_checkbox.isChecked():
            self.scale_y_spinbox.setValue(self.scale_x_spinbox.value())
            self.scale_x_spinbox.valueChanged.connect(
                lambda: self.scale_y_spinbox.setValue(self.scale_x_spinbox.value()))
        else:
            try:
                self.scale_x_spinbox.valueChanged.disconnect()
            except TypeError:
                pass

    #MODERN YÖNTEMLER
    def create_modern_tab(self):
        """Create the modern methods tab content"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Example modern techniques (students will implement these)
        groups = [
            ("Enhancement", ["Histogram Equalization", "Contrast Stretching"]),
            ("Segmentation", ["Threshold", "K-means", "Watershed"]),
            ("Feature Extraction", ["SIFT", "SURF", "ORB"])
        ]

        for i, (group_name, methods) in enumerate(groups):
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()

            for method in methods:
                #SIFT
                if method == "SIFT":
                    self.sift_checkbox = QCheckBox("SIFT")
                    self.sift_checkbox.setEnabled(True)
                    sift_row_layout = QHBoxLayout()
                    sift_row_layout.addWidget(self.sift_checkbox)

                    # Contrast Threshold
                    contrast_label = QLabel("ContrastThres:")
                    bold_font = contrast_label.font()
                    bold_font.setBold(True)
                    contrast_label.setFont(bold_font)
                    self.sift_contrast_slider = QSlider(Qt.Orientation.Horizontal)
                    self.sift_contrast_slider.setRange(1, 100)  # 0.01 - 1.00
                    self.sift_contrast_slider.setValue(40)
                    self.sift_contrast_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.sift_contrast_slider.setTickInterval(10)
                    self.sift_contrast_slider.setFixedWidth(100)

                    # Edge Threshold
                    edge_label = QLabel("EdgeThres:")
                    edge_label.setFont(bold_font)
                    self.sift_edge_slider = QSlider(Qt.Orientation.Horizontal)
                    self.sift_edge_slider.setRange(1, 50)
                    self.sift_edge_slider.setValue(10)
                    self.sift_edge_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.sift_edge_slider.setTickInterval(5)
                    self.sift_edge_slider.setFixedWidth(100)
                    sift_row_layout.addWidget(contrast_label)
                    sift_row_layout.addWidget(self.sift_contrast_slider)
                    sift_row_layout.addWidget(edge_label)
                    sift_row_layout.addWidget(self.sift_edge_slider)
                    group_layout.addLayout(sift_row_layout)

                elif method == "SURF":
                    self.surf_checkbox = QCheckBox("SURF")
                    self.surf_checkbox.setEnabled(True)
                    surf_row_layout = QHBoxLayout()
                    surf_row_layout.addWidget(self.surf_checkbox)

                    # Hessian Threshold slider
                    hessian_label = QLabel("HessianThres:")
                    bold_font = hessian_label.font()
                    bold_font.setBold(True)
                    hessian_label.setFont(bold_font)
                    self.surf_hessian_slider = QSlider(Qt.Orientation.Horizontal)
                    self.surf_hessian_slider.setRange(100, 10000)
                    self.surf_hessian_slider.setValue(400)
                    self.surf_hessian_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.surf_hessian_slider.setTickInterval(500)
                    self.surf_hessian_slider.setFixedWidth(120)
                    surf_row_layout.addWidget(hessian_label)
                    surf_row_layout.addWidget(self.surf_hessian_slider)
                    group_layout.addLayout(surf_row_layout)

                elif method == "ORB":
                    self.modern_orb_checkbox = QCheckBox("ORB")
                    self.modern_orb_checkbox.setEnabled(True)
                    orb_row_layout = QHBoxLayout()
                    orb_row_layout.addWidget(self.modern_orb_checkbox)

                    bold_font = QLabel().font()
                    bold_font.setBold(True)
                    # Edge Threshold Slider
                    edge_label = QLabel("EdgeThres:")
                    edge_label.setFont(bold_font)
                    self.orb_edge_slider = QSlider(Qt.Orientation.Horizontal)
                    self.orb_edge_slider.setRange(1, 50)
                    self.orb_edge_slider.setValue(15)
                    self.orb_edge_slider.setFixedWidth(90)
                    self.orb_edge_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.orb_edge_slider.setTickInterval(5)

                    # Fast Threshold Slider
                    fast_label = QLabel("FastThres:")
                    fast_label.setFont(bold_font)
                    self.orb_fast_slider = QSlider(Qt.Orientation.Horizontal)
                    self.orb_fast_slider.setRange(1, 100)
                    self.orb_fast_slider.setValue(20)
                    self.orb_fast_slider.setFixedWidth(90)
                    self.orb_fast_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.orb_fast_slider.setTickInterval(10)
                    orb_row_layout.addWidget(edge_label)
                    orb_row_layout.addWidget(self.orb_edge_slider)
                    orb_row_layout.addWidget(fast_label)
                    orb_row_layout.addWidget(self.orb_fast_slider)
                    group_layout.addLayout(orb_row_layout)

                elif method == "Histogram Equalization":
                    self.hist_eq_checkbox = QCheckBox("Histogram Equalization")
                    self.hist_eq_checkbox.setEnabled(True)
                    group_layout.addWidget(self.hist_eq_checkbox)

                elif method == "Contrast Stretching":
                    self.contrast_stretch_checkbox = QCheckBox("Contrast Stretching")
                    self.contrast_stretch_checkbox.setEnabled(True)
                    group_layout.addWidget(self.contrast_stretch_checkbox)

                elif method == "Threshold":
                    self.threshold_checkbox = QCheckBox("Threshold")
                    self.threshold_checkbox.setEnabled(True)

                    threshold_row_layout = QHBoxLayout()
                    threshold_row_layout.addWidget(self.threshold_checkbox)

                    threshold_label = QLabel("Threshold:")
                    bold_font = threshold_label.font()
                    bold_font.setBold(True)
                    threshold_label.setFont(bold_font)

                    self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
                    self.threshold_slider.setRange(0, 255)
                    self.threshold_slider.setValue(127)
                    self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.threshold_slider.setTickInterval(25)
                    self.threshold_slider.setFixedWidth(120)

                    threshold_row_layout.addWidget(threshold_label)
                    threshold_row_layout.addWidget(self.threshold_slider)

                    group_layout.addLayout(threshold_row_layout)

                elif method == "K-means":
                    self.kmeans_checkbox = QCheckBox("K-means")
                    self.kmeans_checkbox.setEnabled(True)

                    kmeans_row_layout = QHBoxLayout()
                    kmeans_row_layout.addWidget(self.kmeans_checkbox)

                    k_label = QLabel("K:")
                    bold_font = k_label.font()
                    bold_font.setBold(True)
                    k_label.setFont(bold_font)

                    self.kmeans_k_slider = QSlider(Qt.Orientation.Horizontal)
                    self.kmeans_k_slider.setRange(2, 10)
                    self.kmeans_k_slider.setValue(3)
                    self.kmeans_k_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.kmeans_k_slider.setTickInterval(1)
                    self.kmeans_k_slider.setFixedWidth(120)

                    kmeans_row_layout.addWidget(k_label)
                    kmeans_row_layout.addWidget(self.kmeans_k_slider)

                    group_layout.addLayout(kmeans_row_layout)

                elif method == "Watershed":
                    self.watershed_checkbox = QCheckBox("Watershed")
                    self.watershed_checkbox.setEnabled(True)

                    watershed_row_layout = QHBoxLayout()
                    watershed_row_layout.addWidget(self.watershed_checkbox)

                    marker_label = QLabel("MarkerThres:")
                    bold_font = marker_label.font()
                    bold_font.setBold(True)
                    marker_label.setFont(bold_font)

                    self.watershed_marker_slider = QSlider(Qt.Orientation.Horizontal)
                    self.watershed_marker_slider.setRange(0, 255)
                    self.watershed_marker_slider.setValue(50)
                    self.watershed_marker_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                    self.watershed_marker_slider.setTickInterval(25)
                    self.watershed_marker_slider.setFixedWidth(120)

                    watershed_row_layout.addWidget(marker_label)
                    watershed_row_layout.addWidget(self.watershed_marker_slider)

                    group_layout.addLayout(watershed_row_layout)


                else:
                    checkbox = QCheckBox(method)
                    checkbox.setEnabled(False)  # Initially disabled
                    group_layout.addWidget(checkbox)

            group.setLayout(group_layout)
            layout.addWidget(group, i // 2, i % 2)

        return widget

    # DERİN ÖĞRENME YÖNTEMLERİ
    def create_deep_learning_tab(self):
        """Create the Deep Learning methods tab content"""
        widget = QWidget()
        layout = QGridLayout(widget)

        groups = [
            ("Classification", ["ResNet"]),
            ("Detection", ["YOLOv8"]),
            ("Segmentation", ["U-Net"]),
            ("Anomaly Detection", ["PADIM"]),
            ("Style Transfer", ["Van Gogh"])
        ]

        for i, (group_name, methods) in enumerate(groups):
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()

            # Model seçim dropdown
            combo = QComboBox()
            combo.addItems(methods)

            # Process button
            process_button = QPushButton("Run Inference")
            process_button.setObjectName(group_name.lower().replace(" ", "_") + "_btn")

            if group_name == "Classification":
                process_button.clicked.connect(self.run_classification)

            elif group_name == "Detection":
                process_button.clicked.connect(self.run_detection)

                # Confidence threshold slider
                threshold_layout = QHBoxLayout()
                threshold_label = QLabel("Confidence Threshold:")
                threshold_spin = QDoubleSpinBox()
                threshold_spin.setRange(0.0, 1.0)
                threshold_spin.setSingleStep(0.05)
                threshold_spin.setValue(self.confidence_threshold)
                threshold_spin.valueChanged.connect(lambda val: setattr(self, 'confidence_threshold', val))

                threshold_layout.addWidget(threshold_label)
                threshold_layout.addWidget(threshold_spin)
                group_layout.addLayout(threshold_layout)

            if group_name == "Segmentation":
                process_button.clicked.connect(self.run_segmentation)

                # Threshold ayarı
                threshold_layout = QHBoxLayout()
                threshold_label = QLabel("Confidence Threshold:")
                threshold_spin = QDoubleSpinBox()
                threshold_spin.setRange(0.0, 1.0)
                threshold_spin.setSingleStep(0.05)
                threshold_spin.setValue(self.segmentation_threshold)
                threshold_spin.valueChanged.connect(lambda val: setattr(self, 'segmentation_threshold', val))

                threshold_layout.addWidget(threshold_label)
                threshold_layout.addWidget(threshold_spin)
                group_layout.addLayout(threshold_layout)

            elif group_name == "Anomaly Detection":
                process_button.clicked.connect(self.run_padim_inference)

            elif group_name == "Style Transfer":
                process_button.clicked.connect(self.run_style_transfer)

            # Store references
            setattr(self, f"{group_name.lower().replace(' ', '_')}_combo", combo)
            setattr(self, f"{group_name.lower().replace(' ', '_')}_button", process_button)

            group_layout.addWidget(combo)
            group_layout.addWidget(process_button)
            group.setLayout(group_layout)

            layout.addWidget(group, i // 2, i % 2)

        return widget

    def run_classification(self):
        if self.current_image is None:
            self.show_error("Please load an image first.")
            return

        model_name = self.classification_combo.currentText()

        try:
            if self.classification_model is None or self.loaded_model_name != model_name:
                if model_name == "ResNet":
                    self.classification_model = models.resnet18(pretrained=True)
                elif model_name == "VGG":
                    self.classification_model = models.vgg16(pretrained=True)
                elif model_name == "MobileNet":
                    self.classification_model = models.mobilenet_v2(pretrained=True)
                else:
                    self.show_error("Model not supported.")
                    return
                self.classification_model.eval()
                self.loaded_model_name = model_name

            img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            input_tensor = self.classification_transform(img).unsqueeze(0)

            with torch.no_grad():
                output = self.classification_model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()

            label = self.imagenet_classes[prediction] if prediction < len(
                self.imagenet_classes) else f"Class {prediction}"
            self.status_bar.showMessage(f"{model_name} Prediction: {label} (class {prediction})")

        except Exception as e:
            self.show_error(f"Error running classification: {str(e)}")

    def run_detection(self):
        model_name = self.detection_combo.currentText()

        if self.source_combo.currentText() in ["Video File", "Webcam"]:
            self.yolo_inference_active = True
            self.status_bar.showMessage("YOLOv8 Live Inference: ON")
            return

        if self.current_image is None:
            self.show_error("Please load an image first.")
            return

        try:
            if self.yolo_model is None:
                if model_name == "YOLOv8":
                    self.yolo_model = YOLO("yolov8n.pt")
                else:
                    self.show_error(f"Model {model_name} not supported yet.")
                    return

            image = self.current_image.copy()
            results = self.yolo_model(image, verbose=False)[0]

            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            class_names = self.yolo_model.names  # COCO sınıf isimleri

            for (x1, y1, x2, y2), conf, cls_idx in zip(boxes, confs, classes):
                label = f"{class_names[cls_idx]}: {conf:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.processed_image = image
            self.display_image(self.processed_image, self.output_image_label)
            self.status_bar.showMessage(f"{model_name} detection complete with {len(boxes)} objects.")

        except Exception as e:
            self.show_error(f"YOLOv8 Error: {str(e)}")

    def run_segmentation(self):
        if self.current_image is None:
            self.show_error("Please load an image first.")
            return

        model_name = self.segmentation_combo.currentText()

        try:
            if self.segmentation_model is None or self.loaded_segmentation_model_name != model_name:
                if model_name == "U-Net":
                    from unet import UNet
                    self.segmentation_model = UNet(n_channels=3, n_classes=2)
                    self.segmentation_model.load_state_dict(
                        torch.load("unet_carvana_cpu.pth", map_location="cpu")
                    )
                    self.segmentation_model.eval()
                    self.loaded_segmentation_model_name = model_name
                else:
                    self.show_error("Segmentation model not supported.")
                    return

            img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            input_tensor = self.segmentation_transform(img).unsqueeze(0)

            with torch.no_grad():
                output = self.segmentation_model(input_tensor)[0][0]
                mask = (output > self.segmentation_threshold).float().numpy()

            # Orijinal görüntü boyutuna resize
            mask = cv2.resize(mask, (self.current_image.shape[1], self.current_image.shape[0]))

            # Maske resmi oluştur
            mask_color = np.zeros_like(self.current_image)
            mask_color[:, :, 1] = (mask * 255).astype(np.uint8)  # yeşil kanal
            result = cv2.addWeighted(self.current_image, 0.7, mask_color, 0.3, 0)

            self.processed_image = result
            self.display_image(self.processed_image, self.output_image_label)
            self.status_bar.showMessage(f"{model_name} segmentation complete.")

        except Exception as e:
            self.show_error(f"Segmentation Error: {str(e)}")

    def run_padim_inference(self):
        if self.current_image is None:
            self.show_error("Please load an image first.")
            return

        try:
            from PIL import Image  # Hata almamak için

            # OpenCV → PIL formatına çevir ve boyutlandır
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image).resize((256, 256))
            image_tensor = transforms.ToTensor()(pil_image).unsqueeze(0)

            # Model yükle
            if not hasattr(self, 'padim_model') or self.padim_model is None:
                self.padim_model = load_model("padim_bottle.pth")

            # PaDiM inference
            score, mask = infer(self.padim_model, image_tensor)

            # Maskeyi normalize et ve yeniden boyutlandır
            mask_resized = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]))
            heatmap_resized = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)

            # Isı haritasını orijinal görüntüyle harmanla
            blended = cv2.addWeighted(rgb_image, 0.6, heatmap_resized, 0.8, 1)

            # Görseli göster
            self.processed_image = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            self.display_image(self.processed_image, self.output_image_label)
            self.status_bar.showMessage(f"PaDiM Anomaly Score: {score:.4f}")


        except Exception as e:
            self.show_error(f"PaDiM Inference Error: {str(e)}")

    def run_style_transfer(self):
        if self.current_image is None:
            self.show_error("Please load an image first.")
            return

        try:
            from style_transfer.models.transformer_net import TransformerNet

            model = TransformerNet()
            model_path = "style_transfer/models/vangogh.pth"
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            # Görüntüyü hazırla
            image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image).resize((256, 256))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            input_tensor = transform(image).unsqueeze(0)

            # İnfer
            with torch.no_grad():
                output_tensor = model(input_tensor).clamp(0, 255)

            # NumPy formata çevir
            output_image = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype("uint8")
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            self.processed_image = output_image
            self.display_image(self.processed_image, self.output_image_label)
            self.status_bar.showMessage("Van Gogh style transfer complete.")

        except Exception as e:
            self.show_error(f"Style transfer error: {str(e)}")

    def create_control_panel(self):
        """Create the bottom control panel."""
        control_group = QGroupBox("Processing Controls")
        control_layout = QHBoxLayout()

        # Processing options
        self.additive_checkbox = QCheckBox("Additive Operations")
        self.live_preview_checkbox = QCheckBox("Live Preview")

        # Control buttons
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_image)

        self.preview_controls_btn = QPushButton("View Active Filters")
        self.preview_controls_btn.clicked.connect(self.open_processing_control)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_video)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_processing)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_last_operation)

        # Add widgets to layout
        control_layout.addWidget(self.additive_checkbox)
        control_layout.addWidget(self.live_preview_checkbox)
        control_layout.addStretch()
        control_layout.addWidget(self.undo_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.preview_controls_btn)
        control_layout.addWidget(self.reset_btn)

        control_group.setLayout(control_layout)
        self.main_layout.addWidget(control_group)

    def open_processing_control(self):
        self.processing_control_dialog = ProcessingControlDialog(
            self,
            self.active_filters,
            self.update_active_filters,
            self.process_image  # override_filters argümanını destekliyor
        )
        self.processing_control_dialog.show()

    def set_checkbox_state(self, name, state):
        checkbox_map = {
            "Gaussian Blur": self.gaussian_checkbox,
            "Median Filter": self.median_checkbox,
            "Bilateral Filter": self.bilateral_checkbox,
            "Sobel": self.sobel_checkbox,
            "Canny": self.canny_checkbox,
            "Laplacian": self.laplacian_checkbox,
            "Shi-Tomasi": self.shi_tomasi_checkbox,
            "Prewitt": self.prewitt_checkbox,
            "ORB": self.orb_checkbox,
            "FAST": self.fast_checkbox,
            "Erosion": self.erosion_checkbox,
            "Dilation": self.dilation_checkbox,
            "Opening": self.opening_checkbox,
            "Closing": self.closing_checkbox,
            "Hough Lines": self.hough_lines_checkbox,
            "Hough Circles": self.hough_circles_checkbox,
            "RGB": self.rgb_checkbox,
            "HSV": self.hsv_checkbox,
            "BGR": self.bgr_checkbox,
            "Grayscale": self.grayscale_checkbox,
            "Resize": self.resize_checkbox,
            "Rotate": self.rotate_checkbox,
            "Flip": self.flip_checkbox,
            "Translation": self.translation_checkbox,
            "Affine": self.affine_checkbox,
            "SIFT": self.sift_checkbox,
        }

        if name in checkbox_map:
            checkbox_map[name].setChecked(state)

    def update_active_filters(self, updated_filters):
        # Sadece geçici olarak devre dışı bırakılanları ayıkla
        self.disabled_filters = {name for name, is_active in updated_filters.items() if not is_active}

        # GUI checkbox'larını güncelle
        for name, is_active in updated_filters.items():
            if name in self.active_operations:
                self.set_checkbox_state(name, is_active)

        self.process_image()  # İşleme tekrar uygula

    def change_color_space(self):
        """Store the selected color space, do not apply transformation immediately"""
        self.selected_color_space = self.color_space_combo.currentText()

    def pause_video(self):
        """Pause or resume video playback"""
        if self.video_capture is not None and self.video_capture.isOpened():
            if self.timer.isActive():
                # Video/webcam güncellemesini durdur
                self.timer.stop()
                if self.current_image is not None:
                    self.processed_image = self.current_image.copy()  # Durdurulan kareyi al
                    self.display_image(self.processed_image, self.output_image_label)
                    self.status_bar.showMessage("Video paused. You can now process the current frame.")

                self.pause_btn.setText("Continue")  # Buton metnini değiştir

            else:
                # Video kaldığı yerden devam etsin
                self.timer.start(30)
                self.status_bar.showMessage("Video resumed.")
                self.pause_btn.setText("Pause")  # Buton metnini geri değiştir
        else:
            self.show_error("No video or webcam is active!")

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Add image info label
        self.image_info_label = QLabel()
        self.status_bar.addPermanentWidget(self.image_info_label)

    def add_toolbar_button(self, toolbar, text, slot):
        """Helper method to add buttons to toolbar"""
        button = QPushButton(text)
        button.clicked.connect(slot)
        toolbar.addWidget(button)

    def select_source(self):
        """Handle source selection based on combo box choice"""
        source_type = self.source_combo.currentText()
        try:
            if source_type == "Single Image":
                self.open_image()
            elif source_type == "Video File":
                self.open_video()
            elif source_type == "Webcam":
                self.start_webcam()
        except Exception as e:
            self.show_error(f"Error selecting source: {str(e)}")

    def open_image(self):
        """Open and load an image file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_name:
            try:
                self.load_image(file_name)
                self.status_bar.showMessage(f"Loaded image: {os.path.basename(file_name)}")
            except Exception as e:
                self.show_error(f"Error loading image: {str(e)}")

    def open_video(self):
        """Open and load a video file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_name:
            try:
                self.start_video(file_name)
                self.status_bar.showMessage(f"Playing video: {os.path.basename(file_name)}")
            except Exception as e:
                self.show_error(f"Error loading video: {str(e)}")

    def start_webcam(self):
        """Initialize and start webcam capture"""
        try:
            self.video_capture = cv2.VideoCapture(0)
            if self.video_capture.isOpened():
                # Set resolution based on combo box
                resolution = self.resolution_combo.currentText()
                width, height = map(int, resolution.split('x'))
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                self.timer.start(30)  # 30ms refresh rate
                self.status_bar.showMessage("Webcam started")
            else:
                raise Exception("Could not open webcam")
        except Exception as e:
            self.show_error(f"Error starting webcam: {str(e)}")

    def change_source(self):
        """Handle source type change"""
        if self.video_capture is not None:
            self.video_capture.release()
            self.timer.stop()
        self.reset_processing()
        self.resolution_combo.setVisible(self.source_combo.currentText() == "Webcam")

    def load_image(self, file_name):
        """Load and display an image file"""
        self.current_image = cv2.imread(file_name)
        if self.current_image is None:
            raise Exception("Could not load image")

        self.processed_image = self.current_image.copy()
        self.original_height, self.original_width = self.current_image.shape[:2]
        self.display_image(self.current_image, self.input_image_label)
        self.display_image(self.current_image, self.output_image_label)
        self.update_image_info()
        self.shi_tomasi_quality_slider.valueChanged.connect(self.on_shi_tomasi_slider_change)

    def start_video(self, file_name):
        """Start video playback"""
        self.video_capture = cv2.VideoCapture(file_name)
        if not self.video_capture.isOpened():
            raise Exception("Could not open video file")
        self.timer.start(30)

    def update_frame(self):
        """Update frame for video/webcam display"""
        ret, frame = self.video_capture.read()
        if ret:
            self.current_image = frame
            self.processed_image = frame.copy()
            self.display_image(frame, self.input_image_label)

            if self.live_preview_checkbox.isChecked():
                self.process_image()

            elif self.yolo_inference_active:
                image = frame.copy()
                results = self.yolo_model(image, verbose=False)[0]

                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                class_names = self.yolo_model.names

                for (x1, y1, x2, y2), conf, cls_idx in zip(boxes, confs, classes):
                    if conf < self.confidence_threshold:
                        continue

                    label = f"{class_names[cls_idx]}: {conf:.2f}"
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self.processed_image = image
                self.display_image(self.processed_image, self.output_image_label)

            else:
                self.display_image(frame, self.output_image_label)

            self.update_image_info()
        else:
            # Video ended or frame grab failed
            self.timer.stop()
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            self.status_bar.showMessage("Video ended")

    def display_image(self, image, label):
        """Display an image on a QLabel with proper scaling"""
        if image is not None:
            height, width = image.shape[:2]
            label_width = label.width()
            label_height = label.height()

            # Calculate aspect ratio preserving scaling
            aspect_ratio = width / height
            if label_width / aspect_ratio <= label_height:
                new_width = label_width
                new_height = int(label_width / aspect_ratio)
            else:
                new_height = label_height
                new_width = int(label_height * aspect_ratio)

            # !!! Sadece input_image_label için scale_x ve scale_y ayarla !!!
            if label == self.input_image_label:
                self.scale_x = new_width / width
                self.scale_y = new_height / height

            # Convert the image to RGB format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # Create QImage and QPixmap
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(new_width, new_height,
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled_pixmap)

            # Aynı görseli static QLabel'lara da ata (scroll kapalıyken görünmesi için)
            if label == self.input_image_label and hasattr(self, "input_static_label"):
                self.input_static_label.setPixmap(scaled_pixmap)
            elif label == self.output_image_label and hasattr(self, "output_static_label"):
                self.output_static_label.setPixmap(scaled_pixmap)

    @pyqtSlot()
    def process_image(self, override_filters=None):
        self.active_filters = {}

        if self.epipolar_checkbox.isChecked():
            if self.epipolar_left_image is None or self.epipolar_right_image is None:
                self.show_error("Please load both left and right images for epipolar processing.")
                return
            # Epipolar işleminde processed_image olmadan devam edilebilir
        else:
            if self.current_image is None:
                self.show_error("No image loaded")
                return
            if self.processed_image is None:
                self.processed_image = self.current_image.copy()

        try:
            if override_filters is not None:
                filters_to_use = override_filters
            else:
                filters_to_use = {
                    "Gaussian Blur": self.gaussian_checkbox.isChecked(),
                    "Median Filter": self.median_checkbox.isChecked(),
                    "Bilateral Filter": self.bilateral_checkbox.isChecked(),
                    "Sobel": self.sobel_checkbox.isChecked(),
                    "Canny": self.canny_checkbox.isChecked(),
                    "Laplacian": self.laplacian_checkbox.isChecked(),
                    "Shi-Tomasi": self.shi_tomasi_checkbox.isChecked(),
                    "Prewitt": self.prewitt_checkbox.isChecked(),
                    "ORB": self.orb_checkbox.isChecked(),
                    "FAST": self.fast_checkbox.isChecked(),
                    "Erosion": self.erosion_checkbox.isChecked(),
                    "Dilation": self.dilation_checkbox.isChecked(),
                    "Opening": self.opening_checkbox.isChecked(),
                    "Closing": self.closing_checkbox.isChecked(),
                    "Hough Lines": self.hough_lines_checkbox.isChecked(),
                    "Hough Circles": self.hough_circles_checkbox.isChecked(),
                    "RGB": self.rgb_checkbox.isChecked(),
                    "HSV": self.hsv_checkbox.isChecked(),
                    "BGR": self.bgr_checkbox.isChecked(),
                    "Grayscale": self.grayscale_checkbox.isChecked()
                }
            filters_to_use = {
                k: v for k, v in filters_to_use.items()
                if k not in self.disabled_filters
            }

            self.active_operations = filters_to_use

            self.active_operations = {
                "Gaussian Blur": self.gaussian_checkbox.isChecked(),
                "Median Filter": self.median_checkbox.isChecked(),
                "Bilateral Filter": self.bilateral_checkbox.isChecked(),
                "Sobel": self.sobel_checkbox.isChecked(),
                "Canny": self.canny_checkbox.isChecked(),
                "Laplacian": self.laplacian_checkbox.isChecked(),
                "Shi-Tomasi": self.shi_tomasi_checkbox.isChecked(),
                "Prewitt": self.prewitt_checkbox.isChecked(),
                "ORB": self.orb_checkbox.isChecked(),
                "FAST": self.fast_checkbox.isChecked(),
                "Erosion": self.erosion_checkbox.isChecked(),
                "Dilation": self.dilation_checkbox.isChecked(),
                "Opening": self.opening_checkbox.isChecked(),
                "Closing": self.closing_checkbox.isChecked(),
                "Hough Lines": self.hough_lines_checkbox.isChecked(),
                "Hough Circles": self.hough_circles_checkbox.isChecked(),
                "RGB": self.rgb_checkbox.isChecked(),
                "HSV": self.hsv_checkbox.isChecked(),
                "BGR": self.bgr_checkbox.isChecked(),
                "Grayscale": self.grayscale_checkbox.isChecked()
            }

            # İşlem geçmişini kaydet (geri alma işlemi için)
            if self.processed_image is not None:
                self.processing_history.append(self.processed_image.copy())

            # Additive Operations aktif değilse, orijinal görüntüyü kullan
            if not self.additive_checkbox.isChecked():
                self.processed_image = self.current_image.copy()

            is_grayscale = False

            # Renk Uzayı Dönüşümleri
            if self.bgr_checkbox.isChecked():
                if len(self.processed_image.shape) == 3:
                    self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                    self.active_filters["BGR"] = True

            elif self.hsv_checkbox.isChecked():
                if len(self.processed_image.shape) == 3:
                    self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2HSV)
                    self.active_filters["HSV"] = True

            elif self.rgb_checkbox.isChecked():
                if len(self.processed_image.shape) == 3:
                    pass  # RGB dönüşümü gerekli değil
                    self.active_filters["RGB"] = True

            elif self.grayscale_checkbox.isChecked():
                if len(self.processed_image.shape) == 3:
                    self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                    is_grayscale = True
                    self.active_filters["Grayscale"] = True

            # Gaussian Blur Uygulama
            if self.gaussian_checkbox.isChecked():
                kernel_size = self.gaussian_slider.value()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                sigma_value = self.gaussian_sigma_slider.value()
                self.processed_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), sigma_value)
                self.active_filters["Gaussian Blur"] = True

            # Median Filter Uygulama
            if self.median_checkbox.isChecked():
                kernel_size = self.median_slider.value()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                self.processed_image = cv2.medianBlur(self.processed_image, kernel_size)
                self.active_filters["Median Filter"] = True

            #Bilateral
            if self.bilateral_checkbox.isChecked():
                kernel_size = self.bilateral_kernel_slider.value()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                sigma_color = self.bilateral_slider.value()
                self.processed_image = cv2.bilateralFilter(self.processed_image, kernel_size, sigma_color, 75)
                self.active_filters["Bilateral Filter"] = True

            # Kenar ve Köşe Algılama
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY) if not is_grayscale else self.processed_image
            if self.sobel_checkbox.isChecked():
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                direction = self.sobel_direction_combo.currentText()
                dx, dy = 1, 0  # Default: X
                if direction == "Y":
                    dx, dy = 0, 1
                elif direction == "Both":
                    dx, dy = 1, 1

                sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=5)
                self.processed_image = cv2.convertScaleAbs(sobel)
                self.active_filters["Sobel Edge Detection"] = True

            #Shi-Tomasi
            if self.shi_tomasi_checkbox.isChecked():
                if self.shi_tomasi_quality_slider is not None:
                    quality_val = self.shi_tomasi_quality_slider.value() / 100.0
                else:
                    quality_val = 0.1  # varsayılan fallback

                print(f"[DEBUG] Shi-Tomasi Quality Level: {quality_val:.2f}")

                corners = cv2.goodFeaturesToTrack(gray, 200, quality_val, 10)
                if corners is not None:
                    print(f"[DEBUG] Number of corners: {len(corners)}")
                    for corner in corners.astype(int):  # np.int0 yerine
                        x, y = corner.ravel()
                        cv2.circle(self.processed_image, (x, y), 4, (0, 255, 255), -1)
                else:
                    print("[DEBUG] No corners detected.")

                self.active_filters["Shi-Tomasi"] = True

            if self.canny_checkbox.isChecked():
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)

                # Kernel değeri (3, 5, 7)
                kernel_options = [3, 5, 7]
                kernel_size = kernel_options[self.canny_kernel_slider.value()]

                # Kullanıcının girdiği L.Thres değeri
                lower_thresh = self.canny_lthres_spinbox.value()

                # Üst threshold
                upper_thresh = self.canny_uthres_spinbox.value()

                self.processed_image = cv2.Canny(gray, lower_thresh, upper_thresh, apertureSize=kernel_size)
                self.active_filters["Canny Edge Detection"] = True

            if self.fast_checkbox.isChecked():
                threshold = self.fast_quality_slider.value()
                fast = cv2.FastFeatureDetector_create(threshold=threshold)
                keypoints = fast.detect(gray, None)
                self.processed_image = cv2.drawKeypoints(self.processed_image, keypoints, None, color=(255, 0, 0))
                self.active_filters["FAST"] = True

            if self.laplacian_checkbox.isChecked():
                ksize = [1, 3, 5, 7, 9][self.laplacian_kernel_slider.value()]
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                self.processed_image = cv2.convertScaleAbs(laplacian)
                self.active_filters["Laplacian"] = True

            if self.prewitt_checkbox.isChecked():
                direction = self.prewitt_direction_combo.currentText()

                # Prewitt çekirdekleri
                kernelx = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]], dtype=np.float32)
                kernely = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]], dtype=np.float32)

                if direction in ["X", "Both"]:
                    grad_x = cv2.filter2D(gray, cv2.CV_64F, kernelx)
                else:
                    grad_x = np.zeros_like(gray, dtype=np.float64)

                if direction in ["Y", "Both"]:
                    grad_y = cv2.filter2D(gray, cv2.CV_64F, kernely)
                else:
                    grad_y = np.zeros_like(gray, dtype=np.float64)

                magnitude = cv2.magnitude(grad_x, grad_y)
                self.processed_image = cv2.convertScaleAbs(magnitude)
                self.active_filters["Prewitt"] = True

            if self.orb_checkbox.isChecked():
                orb_threshold = self.orb_quality_slider.value()
                orb = cv2.ORB_create(fastThreshold=orb_threshold)
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                self.processed_image = cv2.drawKeypoints(self.processed_image, keypoints, None, color=(0, 255, 0))
                self.active_filters["ORB"] = True

            # HOUGH LINES
            if self.hough_lines_checkbox.isChecked():
                gray_lines = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_lines, 50, 150, apertureSize=3)
                rho = self.hough_rho_slider.value()
                rho = self.hough_rho_slider.value()
                theta_deg = self.hough_theta_slider.value()
                theta = np.deg2rad(theta_deg)

                threshold = self.hough_threshold_slider.value()
                lines = cv2.HoughLines(edges, rho, theta, threshold)

                if lines is not None:
                    for rho, theta in lines[:, 0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(self.processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        self.active_filters["Hough Lines"] = True
            # HOUGH CIRCLES
            if self.hough_circles_checkbox.isChecked():
                gray_circles = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                gray_blurred = cv2.medianBlur(gray_circles, 5)

                param1 = self.hough_param1_slider.value()
                param2 = self.hough_param2_slider.value()
                min_radius = self.hough_min_radius_spinbox.value()
                max_radius = self.hough_max_radius_spinbox.value()

                circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                           param1=param1, param2=param2,
                                           minRadius=min_radius, maxRadius=max_radius)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        x, y, r = circle
                        cv2.circle(self.processed_image, (x, y), r, (0, 255, 0), 2)
                        cv2.circle(self.processed_image, (x, y), 2, (0, 0, 255), 3)
                        self.active_filters["Hough Circles"] = True

            # Morfolojik işlemler için çekirdek tanımı
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

            # Sadece morfolojik checkbox'lardan biri seçiliyse işlem yap
            if any([self.erosion_checkbox.isChecked(), self.dilation_checkbox.isChecked(),
                    self.opening_checkbox.isChecked(), self.closing_checkbox.isChecked()]):

                morph_input = self.processed_image
                converted_to_gray = False

                if len(morph_input.shape) == 3:
                    morph_input = cv2.cvtColor(morph_input, cv2.COLOR_BGR2GRAY)
                    converted_to_gray = True

                # Erosion
                if self.erosion_checkbox.isChecked():
                    ksize = self.erosion_kernel_slider.value()
                    if ksize % 2 == 0:
                        ksize += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                    morph_input = cv2.erode(morph_input, kernel, iterations=1)
                    self.active_filters["Erosion"] = True

                # Dilation
                if self.dilation_checkbox.isChecked():
                    ksize = self.dilation_kernel_slider.value()
                    if ksize % 2 == 0:
                        ksize += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                    morph_input = cv2.dilate(morph_input, kernel, iterations=1)
                    self.active_filters["Dilation"] = True

                # Opening
                if self.opening_checkbox.isChecked():
                    ksize = self.opening_kernel_slider.value()
                    if ksize % 2 == 0:
                        ksize += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                    morph_input = cv2.morphologyEx(morph_input, cv2.MORPH_OPEN, kernel)
                    self.active_filters["Opening"] = True

                # Closing
                if self.closing_checkbox.isChecked():
                    ksize = self.closing_kernel_slider.value()
                    if ksize % 2 == 0:
                        ksize += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                    morph_input = cv2.morphologyEx(morph_input, cv2.MORPH_CLOSE, kernel)
                    self.active_filters["Closing"] = True

                self.processed_image = morph_input

            #Resize
            if hasattr(self, "resize_checkbox") and self.resize_checkbox.isChecked():
                target_width = self.resize_width_box.value()
                target_height = self.resize_height_box.value()
                interpolation_text = self.interpolation_combo.currentText()
                if interpolation_text == "Nearest-neighbor":
                    interpolation = cv2.INTER_NEAREST
                elif interpolation_text == "Bilinear":
                    interpolation = cv2.INTER_LINEAR
                elif interpolation_text == "Bicubic":
                    interpolation = cv2.INTER_CUBIC
                else:
                    interpolation = cv2.INTER_AREA  # default fallback
                self.processed_image = cv2.resize(self.processed_image, (target_width, target_height),
                                                  interpolation=interpolation)
                self.active_filters["Resize"] = True

            #Rotate
            if hasattr(self, "rotate_checkbox") and self.rotate_checkbox.isChecked():
                angle = self.rotate_angle_spinbox.value()
                px = self.rotate_pivot_x.value()
                py = self.rotate_pivot_y.value()

                (h, w) = self.processed_image.shape[:2]
                if px == 0 and py == 0:
                    center = (w // 2, h // 2)  # Varsayılan pivot: görüntü merkezi
                else:
                    center = (px, py)

                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                self.processed_image = cv2.warpAffine(self.processed_image, M, (w, h))
                self.active_filters["Rotate"] = True

            # Flip
            if hasattr(self, "flip_checkbox") and self.flip_checkbox.isChecked():
                axis = self.flip_axis_combo.currentText()
                flip_code = {"X": 0, "Y": 1, "Both": -1}.get(axis, 1)
                self.processed_image = cv2.flip(self.processed_image, flip_code)
                self.active_filters["Flip"] = True

            # Translation
            if hasattr(self, "translation_checkbox") and self.translation_checkbox.isChecked():
                tx = self.translation_x_spinbox.value()
                ty = self.translation_y_spinbox.value()
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                h, w = self.processed_image.shape[:2]
                self.processed_image = cv2.warpAffine(self.processed_image, M, (w, h))
                self.active_filters["Translation"] = True

            # Scaling
            if hasattr(self, "scaling_checkbox") and self.scaling_checkbox.isChecked():
                sx = self.scale_x_spinbox.value()
                sy = self.scale_y_spinbox.value()
                h, w = self.processed_image.shape[:2]
                new_w = int(w * sx)
                new_h = int(h * sy)
                self.processed_image = cv2.resize(self.processed_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                self.active_filters["Scaling"] = True

            # Shearing
            if hasattr(self, "shearing_checkbox") and self.shearing_checkbox.isChecked():
                shear_x = self.shear_x_spinbox.value()
                shear_y = self.shear_y_spinbox.value()
                h, w = self.processed_image.shape[:2]
                shear_matrix = np.float32([
                    [1, shear_x, 0],
                    [shear_y, 1, 0]
                ])
                self.processed_image = cv2.warpAffine(self.processed_image, shear_matrix, (w, h))
                self.active_filters["Shearing"] = True

            # Affine
            if hasattr(self, "affine_checkbox") and self.affine_checkbox.isChecked():
                angle = self.affine_angle_spinbox.value()
                tx = self.affine_translation_x.value()
                ty = self.affine_translation_y.value()
                scale = self.affine_scale_spinbox.value()

                h, w = self.processed_image.shape[:2]
                center = (w // 2, h // 2)

                M = cv2.getRotationMatrix2D(center, angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty

                self.processed_image = cv2.warpAffine(self.processed_image, M, (w, h))
                self.active_filters["Affine"] = True

            #6DoF Affine
            if hasattr(self, "affine6dof_checkbox") and self.affine6dof_checkbox.isChecked():
                a11 = self.affine6dof_a11_spinbox.value()
                a12 = self.affine6dof_a12_spinbox.value()
                a13 = self.affine6dof_a13_spinbox.value()
                a21 = self.affine6dof_a21_spinbox.value()
                a22 = self.affine6dof_a22_spinbox.value()
                a23 = self.affine6dof_a23_spinbox.value()

                M = np.array([
                    [a11, a12, a13],
                    [a21, a22, a23]
                ], dtype=np.float32)

                h, w = self.processed_image.shape[:2]
                self.processed_image = cv2.warpAffine(self.processed_image, M, (w, h))
                self.active_filters["6DoF Affine"] = True

            #Perspective
            if hasattr(self, "perspective_checkbox") and self.perspective_checkbox.isChecked():
                if len(self.perspective_src_points) == 4:
                    src = np.array(self.perspective_src_points, dtype="float32")
                    width = 400
                    height = 300
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype="float32")

                    M = cv2.getPerspectiveTransform(src, dst)
                    self.processed_image = cv2.warpPerspective(self.current_image, M, (width, height))
                    self.active_filters["Perspective"] = True

                    # Display and early return!
                    self.display_image(self.processed_image, self.output_image_label)
                    self.status_bar.showMessage("Perspective transform applied.")
                    return

                src = np.array(self.perspective_src_points, dtype="float32")
                width = 400
                height = 300
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(src, dst)
                self.processed_image = cv2.warpPerspective(self.current_image, M, (width, height))
                self.active_filters["Perspective"] = True

            #Warp
            if hasattr(self, "warp_checkbox") and self.warp_checkbox.isChecked():
                values = [box.value() for box in self.warp_matrix_spinboxes]
                M = np.array(values, dtype=np.float32).reshape((3, 3))
                h, w = self.processed_image.shape[:2]
                self.processed_image = cv2.warpPerspective(self.processed_image, M, (w, h))
                self.active_filters["Warp"] = True

            # Corner Detection
            if hasattr(self, "corner_detection_checkbox") and self.corner_detection_checkbox.isChecked():
                # Kalite seviyesi slider'dan alınır, yoksa varsayılan 0.01
                quality_val = 0.01
                if hasattr(self, "corner_quality_slider"):
                    quality_val = self.corner_quality_slider.value() / 100.0
                # goodFeaturesToTrack ile köşe tespiti
                corners = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=100,
                    qualityLevel=quality_val,
                    minDistance=10
                )
                if corners is not None:
                    for corner in corners.astype(int):
                        x, y = corner.ravel()
                        cv2.circle(self.processed_image, (x, y), 3, (255, 0, 0), -1)
                self.active_filters["Corner Detection"] = True

            # Line Detection (Probabilistic Hough Lines)
            if hasattr(self, "line_detection_checkbox") and self.line_detection_checkbox.isChecked():
                gray_lines = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_lines, 50, 150, apertureSize=3)
                threshold = self.line_threshold_slider.value()

                lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                        threshold=threshold, minLineLength=50, maxLineGap=10)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(self.processed_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    self.active_filters["Line Detection"] = True

            # Contour Detection
            if hasattr(self, "contour_checkbox") and self.contour_checkbox.isChecked():
                min_area = self.contour_area_slider.value()
                gray_contour = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_contour, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area >= min_area:
                        cv2.drawContours(self.processed_image, [cnt], 0, (0, 0, 255), 2)
                self.active_filters["Contours"] = True

            #SIFT
            if hasattr(self, "sift_checkbox") and self.sift_checkbox.isChecked():
                contrast_thres = self.sift_contrast_slider.value() / 100.0
                edge_thres = self.sift_edge_slider.value()

                sift = cv2.SIFT_create(contrastThreshold=contrast_thres,
                                       edgeThreshold=edge_thres)
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                self.processed_image = cv2.drawKeypoints(self.processed_image, keypoints, None, color=(255, 255, 0))
                self.active_filters["SIFT"] = True

            #SURF
            if hasattr(self, "surf_checkbox") and self.surf_checkbox.isChecked():
                hessian_thresh = self.surf_hessian_slider.value()
                try:
                    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_thresh)
                    keypoints, descriptors = surf.detectAndCompute(gray, None)
                    self.processed_image = cv2.drawKeypoints(self.processed_image, keypoints, None, color=(0, 255, 255))
                    self.active_filters["SURF"] = True
                except AttributeError:
                    self.show_error("SURF is not available. Make sure opencv-contrib-python is installed.")

            # Modern ORB
            if hasattr(self, "modern_orb_checkbox") and self.modern_orb_checkbox.isChecked():
                edge_thres = self.orb_edge_slider.value()
                fast_thres = self.orb_fast_slider.value()
                orb = cv2.ORB_create(edgeThreshold=edge_thres,
                                     fastThreshold=fast_thres)
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                self.processed_image = cv2.drawKeypoints(self.processed_image, keypoints, None, color=(255, 0, 255))
                self.active_filters["ORB (Modern)"] = True

            # Manual Homography
            if hasattr(self, "manual_homography_checkbox") and self.manual_homography_checkbox.isChecked():
                try:
                    src_pts = []
                    dst_pts = []

                    for s_box, t_box in zip(self.homography_src_inputs, self.homography_dst_inputs):
                        sx, sy = map(float, s_box.text().split(','))
                        tx, ty = map(float, t_box.text().split(','))
                        src_pts.append([sx, sy])
                        dst_pts.append([tx, ty])

                    src_pts = np.array(src_pts, dtype=np.float32)
                    dst_pts = np.array(dst_pts, dtype=np.float32)

                    threshold = self.ransac_threshold_spinbox.value()
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
                    if mask is not None:
                        inliers = np.sum(mask)
                        self.status_bar.showMessage(f"Manual Homography applied | Inliers: {inliers}/4")
                    h, w = self.current_image.shape[:2]
                    self.processed_image = cv2.warpPerspective(self.current_image, H, (w, h))
                    self.homography_matrix_label.setText(f"H:\n{H}")
                    self.active_filters["Manual Homography"] = True

                except Exception as e:
                    self.show_error(f"Homography Error: {str(e)}")

            #Epipolar
            if hasattr(self, "epipolar_checkbox") and self.epipolar_checkbox.isChecked():
                if self.epipolar_left_image is not None and self.epipolar_right_image is not None:
                    gray1 = cv2.cvtColor(self.epipolar_left_image, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(self.epipolar_right_image, cv2.COLOR_BGR2GRAY)

                    orb = cv2.ORB_create()
                    kp1, des1 = orb.detectAndCompute(gray1, None)
                    kp2, des2 = orb.detectAndCompute(gray2, None)

                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                    if len(matches) >= 8:
                        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

                        if F is not None:
                            self.status_bar.showMessage("Fundamental matrix estimated.")
                            print("Fundamental Matrix:\n", F)
                            self.active_filters["Epipolar Geometry"] = True

                            epilines_img = self.epipolar_left_image.copy()
                            for m in matches[:30]:
                                pt = tuple(map(int, kp1[m.queryIdx].pt))
                                cv2.circle(epilines_img, pt, 4, (255, 0, 0), -1)

                            self.epipolar_result_array = epilines_img.copy()
                            self.display_image(epilines_img, self.epipolar_output_label)
                            self.epipolar_result_array = epilines_img.copy()
                            self.status_bar.showMessage("Epipolar matches visualized (first 30 shown).")
                            epilines_img = self.epipolar_right_image.copy()

                            # 1. Noktaları çiz
                            for m in matches[:30]:
                                pt = tuple(map(int, kp2[m.trainIdx].pt))  # Sağ görüntüdeki eşleşen nokta
                                cv2.circle(epilines_img, pt, 4, (255, 0, 0), -1)

                            # 2. Epipolar çizgileri çiz (isteğe bağlı)
                            if self.plot_epipolar_lines_checkbox.isChecked():
                                lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
                                lines2 = lines2.reshape(-1, 3)

                                for r in lines2[:30]:
                                    a, b, c = r
                                    x0, y0 = 0, int(-c / b) if b != 0 else 0
                                    x1 = epilines_img.shape[1]
                                    y1 = int(-(a * x1 + c) / b) if b != 0 else 0
                                    cv2.line(epilines_img, (x0, y0), (x1, y1), (0, 255, 0), 1)

                            # 3. Göster
                            self.epipolar_result_array = epilines_img
                            self.display_image(epilines_img, self.epipolar_output_label)

                            self.status_bar.showMessage("Epipolar lines drawn on output image.")

            # Combine using Homography
            if hasattr(self, "combine_homography_checkbox") and self.combine_homography_checkbox.isChecked():
                if hasattr(self, "combine_left_image") and hasattr(self, "combine_right_image"):
                    try:
                        img1 = self.combine_left_image
                        img2 = self.combine_right_image

                        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                        orb = cv2.ORB_create()
                        kp1, des1 = orb.detectAndCompute(gray1, None)
                        kp2, des2 = orb.detectAndCompute(gray2, None)

                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = matcher.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)

                        if len(matches) > 4:
                            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                            if H is not None:
                                h1, w1 = img1.shape[:2]
                                h2, w2 = img2.shape[:2]
                                panorama = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
                                panorama[0:h2, 0:w2] = img2

                                self.processed_image = panorama
                                self.display_image(self.processed_image, self.output_image_label)
                                self.status_bar.showMessage("Homography-based image combination applied.")
                                self.active_filters["Homography Combine"] = True
                    except Exception as e:
                        self.show_error(f"Error combining images: {str(e)}")

            # Histogram Equalization
            if hasattr(self, "hist_eq_checkbox") and self.hist_eq_checkbox.isChecked():
                if len(self.processed_image.shape) == 3:
                    gray_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = self.processed_image

                equalized = cv2.equalizeHist(gray_img)
                self.processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
                self.active_filters["Histogram Equalization"] = True

            # Contrast Stretching
            if hasattr(self, "contrast_stretch_checkbox") and self.contrast_stretch_checkbox.isChecked():
                img = self.processed_image
                if len(img.shape) == 3:
                    channels = cv2.split(img)
                    stretched_channels = []
                    for ch in channels:
                        in_min = np.min(ch)
                        in_max = np.max(ch)
                        if in_max - in_min == 0:
                            stretched = ch.copy()  # Bölünme hatası önlemi
                        else:
                            stretched = ((ch - in_min) / (in_max - in_min) * 255).astype(np.uint8)
                        stretched_channels.append(stretched)
                    self.processed_image = cv2.merge(stretched_channels)
                else:
                    in_min = np.min(img)
                    in_max = np.max(img)
                    if in_max - in_min != 0:
                        self.processed_image = ((img - in_min) / (in_max - in_min) * 255).astype(np.uint8)

                self.active_filters["Contrast Stretching"] = True

            # Threshold
            if hasattr(self, "threshold_checkbox") and self.threshold_checkbox.isChecked():
                thresh_value = self.threshold_slider.value()
                img = self.processed_image
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = img
                _, thresh_img = cv2.threshold(gray_img, thresh_value, 255, cv2.THRESH_BINARY)
                self.processed_image = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
                self.active_filters["Threshold"] = True

            # K-means Segmentation
            if hasattr(self, "kmeans_checkbox") and self.kmeans_checkbox.isChecked():
                k = self.kmeans_k_slider.value()
                img = self.processed_image
                Z = img.reshape((-1, 3))
                Z = np.float32(Z)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                center = np.uint8(center)
                result = center[label.flatten()]
                result_image = result.reshape((img.shape))

                self.processed_image = result_image
                self.active_filters["K-means"] = True

            # Watershed Segmentation
            if hasattr(self, "watershed_checkbox") and self.watershed_checkbox.isChecked():
                img = self.processed_image.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                marker_thresh = self.watershed_marker_slider.value()

                # Threshold ve inverse binary
                _, thresh = cv2.threshold(gray, marker_thresh, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Gürültü temizleme (morph)
                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

                # Arka plan
                sure_bg = cv2.dilate(opening, kernel, iterations=3)

                # Ön plan (mesafe dönüşümü)
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

                # Marker tanımı
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0

                # Watershed uygulama
                markers = cv2.watershed(img, markers)
                img[markers == -1] = [0, 0, 255]  # Kenarları kırmızı çiz

                self.processed_image = img
                self.active_filters["Watershed"] = True

            # Sonuç Görüntüsünü Göster
            self.display_image(self.processed_image, self.output_image_label)
            self.status_bar.showMessage("Processing complete")

        except Exception as e:
            self.show_error(f"Error processing image: {str(e)}")

    def on_corner_slider_change(self):
        if self.is_image_loaded():
            self.process_image()

    def reset_processing(self):
        """Reset the processed image to the original"""
        if self.current_image is not None:
            self.processed_image = self.current_image.copy()
            self.display_image(self.processed_image, self.output_image_label)
            self.processing_history.clear()
            self.perspective_src_points.clear()
            self.status_bar.showMessage("Processing reset")

            # Resize kutularını sıfırla (eğer tanımlıysa)
            if hasattr(self, "resize_width_box") and hasattr(self, "resize_height_box"):
                if self.original_width and self.original_height:
                    self.resize_width_box.setValue(self.original_width)
                    self.resize_height_box.setValue(self.original_height)

    def undo_last_operation(self):
        """Undo the last processing operation"""
        if len(self.processing_history) > 0:
            self.processed_image = self.processing_history.pop()  # Son işlenmiş görüntüyü geri al
            self.display_image(self.processed_image, self.output_image_label)
            self.perspective_src_points.clear()
            self.status_bar.showMessage("Undo last operation")
        else:
            self.show_error("Nothing to undo")

    def save_result(self):
        """Save the processed image and epipolar output image if available."""
        if (
                self.processed_image is None or self.processed_image.size == 0
        ) and not hasattr(self, "epipolar_result_array"):
            self.show_error("No processed or epipolar image to save.")
            return

        # Klasör seç
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not directory:
            return

        try:
            saved_files = []

            # Save processed image
            if self.processed_image is not None and self.processed_image.size != 0:
                output_path = os.path.join(directory, "output_image.png")
                if cv2.imwrite(output_path, self.processed_image):
                    saved_files.append("output_image.png")

            # Save epipolar image from NumPy array if available
            if hasattr(self, "epipolar_result_array") and self.epipolar_result_array is not None:
                epipolar_path = os.path.join(directory, "epipolar_output.png")
                if cv2.imwrite(epipolar_path, self.epipolar_result_array):
                    saved_files.append("epipolar_output.png")

            if saved_files:
                self.status_bar.showMessage(f"Saved: {', '.join(saved_files)}")
            else:
                self.show_error("Nothing was saved. No valid images.")

        except Exception as e:
            self.show_error(f"Error saving results: {str(e)}")

    def export_history(self):
        """Export processing history as a series of images"""
        if not self.processing_history:
            self.show_error("No processing history to export")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if directory:
            try:
                for i, image in enumerate(self.processing_history):
                    filename = os.path.join(directory, f"step_{i + 1}.png")
                    cv2.imwrite(filename, image)
                self.status_bar.showMessage(f"Processing history exported to: {directory}")
            except Exception as e:
                self.show_error(f"Error exporting history: {str(e)}")

    def update_image_info(self):
        """Update image information in status bar"""
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            channels = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            size_mb = self.current_image.nbytes / (1024 * 1024)
            self.image_info_label.setText(
                f"Size: {width}x{height} | Channels: {channels} | Memory: {size_mb:.1f}MB"
            )

    def zoom_in(self):
        """Zoom in on the images"""
        # Implement zoom functionality
        pass

    def zoom_out(self):
        """Zoom out of the images"""
        # Implement zoom functionality
        pass

    def show_error(self, message):
        """Show error message in a dialog box"""
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        """Handle application closing"""
        if self.video_capture is not None:
            self.video_capture.release()
            self.timer.stop()
        event.accept()


def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = ImageProcessingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()