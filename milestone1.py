################################ Includes ##############################################################################

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.uic import loadUi
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

################################ GUI Attachments with functions ##############################################################################

class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('imageprocessinggui.ui', self)
        self.setWindowIcon(QtGui.QIcon("ainshams.png"))
        self.setWindowTitle("Image Augmentation")


        self.image = None
        self.actionOpen.triggered.connect(self.open_img)
        self.actionSave.triggered.connect(self.save_img)
        self.actionBig.triggered.connect(self.big_Img)
        self.actionSmall.triggered.connect(self.small_Img)

        # Image effects 1
        self.actionRotation.triggered.connect(self.rotation)
        self.actionAffine.triggered.connect(self.shearing)
        self.actionTranslation.triggered.connect(self.translation)

        # Image effects 2
        self.actionGrayscale.triggered.connect(self.Grayscale)
        self.actionNegative.triggered.connect(self.Negative)
        self.actionHistogram.triggered.connect(self.histogram_Equalization)
        self.actionLog.triggered.connect(self.Log)
        self.actionGamma.triggered.connect(self.gamma)
        
        # Smoothing
        self.actionBlur.triggered.connect(self.blur)
        self.actionBox_Filter.triggered.connect(self.box_filter)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionBilateral_Filter.triggered.connect(self.bilateral_filter)
        self.actionGaussian_Filter.triggered.connect(self.gaussian_filter)
        
        # Filter
        self.actionMedian_threshold_2.triggered.connect(self.median_threshold)
        self.actionDirectional_Filtering_2.triggered.connect(self.directional_filtering)
        self.actionDirectional_Filtering_3.triggered.connect(self.directional_filtering2)
        self.actionDirectional_Filtering_4.triggered.connect(self.directional_filtering3)
        self.actionMedian_Filtering.triggered.connect(self.median_filtering)

        # Image Noise
        self.actionGaussian.triggered.connect(self.gaussian_noise)
        self.actionUniform.triggered.connect(self.uniform_noise)
        self.actionImpluse.triggered.connect(self.impulse_noise)
        
        # Batch images
        self.actionRotation_Batch.triggered.connect(self.Rotation_batch)
        self.actionTranslation_Batch.triggered.connect(self.Translation_batch)
        self.actionCrop_Batch.triggered.connect(self.Crop_batch)
        self.actionFlip_Batch.triggered.connect(self.Flip_batch)
        
        #Histogram
        self.actionHistogram_PDF.triggered.connect(self.hist)

        # Set input
        self.dial.valueChanged.connect(self.rotation2)
        self.horizontalSlider.valueChanged.connect(self.Gamma_)
        self.gaussian_QSlider.valueChanged.connect(self.gaussian_filter2)
        self.erosion.valueChanged.connect(self.erode)
        self.Qlog.valueChanged.connect(self.Log)
        self.size_Img.valueChanged.connect(self.SIZE)
        self.canny.stateChanged.connect(self.Canny)
        self.canny_min.valueChanged.connect(self.Canny)
        self.canny_max.valueChanged.connect(self.Canny)
        self.pushButton.clicked.connect(self.reset)
        
################################ Basic Functions ##############################################################################
       
    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

        img = img.rgbSwapped() # Convert RGB to BGR
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# align the image with the label
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\DELL\PycharmProjects\DemoPro', "Image Files (*)")
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def save_img(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.image) # save image
        else:
            print("Error")


    def big_Img(self):
        self.image = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def small_Img(self):
        self.image = cv2.resize(self.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def SIZE(self , c):
        self.image = self.tmp
        self.image = cv2.resize(self.image, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def reset(self):
        self.image = self.tmp
        self.displayImage(2)


################################ Image effect 1 ##############################################################################

    def rotation(self):
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def rotation2(self, angle):
        self.image = self.tmp
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def shearing(self):
        #self.image = self.tmp
        rows, cols, ch = self.image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))

        self.displayImage(2)

    def translation(self):
        #self.image = self.tmp
        num_rows, num_cols = self.image.shape[:2]

        translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(self.image, translation_matrix, (num_cols, num_rows))
        self.image = img_translation
        self.displayImage(2)

    def erode(self , iter):
        self.image = self.tmp
        if iter > 0 :
            kernel = np.ones((4, 7), np.uint8)
            self.image = cv2.erode(self.tmp, kernel, iterations=iter)
        else :
            kernel = np.ones((2, 6), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=iter*-1)
        self.displayImage(2)

    def Canny(self):
        self.image = self.tmp
        if self.canny.isChecked():
            can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Canny(can, self.canny_min.value(), self.canny_max.value())
        self.displayImage(2)

################################ Image effect 2 ##############################################################################
   
    def Grayscale(self):
        #self.image = self.tmp
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.displayImage(2)

    def Negative(self):
        #self.image = self.tmp
        self.image = ~self.image
        self.displayImage(2)

    def histogram_Equalization(self):
        #self.image = self.tmp
        img_yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        self.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        self.displayImage(2)

    def Log(self):
        self.image = self.tmp
        img_2 = np.uint8(np.log(self.image))
        c = 2
        self.image = cv2.threshold(img_2, c, 225, cv2.THRESH_BINARY)[1]
        self.displayImage(2)

    def Gamma_(self, gamma):
        #self.image = self.tmp
        gamma = gamma*0.1
        invGamma = 1.0 /gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

    def gamma(self):
        #self.image = self.tmp
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

####################################### Noise ################################################################

    def gaussian_noise(self):
        #self.image = self.tmp
        row, col, ch = self.image.shape
        mean = 0
        var = 0.1
        sigma = var * 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        self.image = self.image + gauss
        self.displayImage(2)
    def uniform_noise(self):
        #self.image = self.tmp
        uniform_noise = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        cv2.randu(uniform_noise, 0, 255)
        self.image = (uniform_noise * 0.5).astype(np.uint8)
        self.displayImage(2)
    def impulse_noise(self):
        #self.image = self.tmp
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(self.image)
        # Salt mode
        num_salt = np.ceil(amount * self.image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in self.image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in self.image.shape]
        out[coords] = 0
        self.image = out
        self.displayImage(2)

####################################### Histogram ################################################################

    def hist(self):
        self.image = self.tmp
        histg = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        self.image = histg
        plt.plot(self.image)
        plt.show()
        #self.displayImage(2)

##################################### Smoothing ##########################################################################

    def blur(self):
        #self.image = self.tmp
        self.image = cv2.blur(self.image, (5, 5))
        self.displayImage(2)
    def box_filter(self):
        self.image = self.tmp
        self.image = cv2.boxFilter(self.image, -1,(20,20))
        self.displayImage(2)
    def median_filter(self):
        #self.image = self.tmp
        self.image = cv2.medianBlur(self.image,5)
        self.displayImage(2)
    def bilateral_filter(self):
        #self.image = self.tmp
        self.image = cv2.bilateralFilter(self.image,9,75,75)
        self.displayImage(2)
    def gaussian_filter(self):
        #self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image,(5,5),0)
        self.displayImage(2)
    def gaussian_filter2(self, g):
        #self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image, (5, 5), g)
        self.displayImage(2)
        
########################################Filter##########################################################################

    def median_threshold(self):
        #self.image = self.tmp
        grayscaled = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.medianBlur(self.image,5)
        retval, threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.image = threshold
        self.displayImage(2)
    def directional_filtering(self):
        #self.image = self.tmp
        kernel = np.ones((3, 3), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)
    def directional_filtering2(self):
        #self.image = self.tmp
        kernel = np.ones((5, 5), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)
    def directional_filtering3(self):
        #self.image = self.tmp
        kernel = np.ones((7, 7), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def median_filtering(self):
            #self.image = self.tmp
            self.image = cv2.medianBlur(self.image, 5)
            self.displayImage(2)


######################################## Batch ##########################################################################

    def Rotation_batch(self):
        rows, cols, steps = self.image.shape
        filenametemplate= "%d.png"
        path = 'F:/ASU/Machine Vision/milestone 1/Batch images'
        for x in range(0,180,20):
            self.image = self.tmp
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), x, 1)
            self.image = cv2.warpAffine(self.image, M, (cols, rows))
            filename= filenametemplate %(x)
            cv2.imwrite(os.path.join(path , filename), self.image)
            
    def Translation_batch(self):
        rows, cols, steps = self.image.shape
        filenametemplate= "%d.png"
        path = 'F:/ASU/Machine Vision/milestone 1/Batch images'
        for x in range(0,110,10):
            self.image = self.tmp
            num_rows, num_cols = self.image.shape[:2]
            translation_matrix = np.float32([[1, 0, x], [0, 1, x+20]])
            img_translation = cv2.warpAffine(self.image, translation_matrix, (num_cols, num_rows))
            self.image = img_translation
            filename= filenametemplate %(x)
            cv2.imwrite(os.path.join(path , filename), self.image)
            
    def Crop_batch(self):
        filenametemplate= "%d.png"
        path = 'F:/ASU/Machine Vision/milestone 1/Batch images'
        img = self.image
        for x in range (0,11):
            cropped_image = img[random.randint(20, 120):random.randint(130, 300), random.randint(100, 150):random.randint(200, 350)]
            filename= filenametemplate %(x)
            cv2.imwrite(os.path.join(path , filename), cropped_image)
    
    def Flip_batch(self):
        filenametemplate= "%d.png"
        path = 'F:/ASU/Machine Vision/milestone 1/Batch images'
        for x in range (-1,2):            	
            flip = cv2.flip(self.image, x)
            filename= filenametemplate %(x+1)
            cv2.imwrite(os.path.join(path , filename), flip)
        
            
######################################## Main Application ##########################################################

app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())

