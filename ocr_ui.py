# -*- coding: utf-8 -*-   
from PyQt4.QtGui import *  
from PyQt4.QtCore import *    
from inferenceGUI import InferenceMain 
  
QTextCodec.setCodecForTr(QTextCodec.codecForName("utf8"))  
  
class LayoutDialog(QDialog):  
    def __init__(self,inferenceChar,parent=None):  
        super(LayoutDialog,self).__init__(parent)  
        self.setWindowTitle(self.tr("线路板批次OCR界面")) 
        self.inferenceChar = inferenceChar
#        font = QFont(self.tr("黑体"),12)
#        QApplication.setFont(font)
        
        mainSplitter=QSplitter(Qt.Vertical,self)
        TopSplitter=QSplitter(Qt.Horizontal,mainSplitter)    
        TopSplitter.setOpaqueResize(False)
        
        leftFrame = QFrame(TopSplitter)
        rightFrame = QFrame(TopSplitter)
        
        leftWindow = LeftDialog()
        self.rightWindow = RightDialog(self.inferenceChar)
        
        leftLayout = QGridLayout(leftFrame)
        leftLayout.addWidget(leftWindow)
        rightLayout = QGridLayout(rightFrame)
        rightLayout.addWidget(self.rightWindow)
        
        BottomFrame = QFrame(mainSplitter)
        self.bottomWindow = BottomDialog(self.inferenceChar)
        bottomLayout = QGridLayout(BottomFrame)
        bottomLayout.addWidget(self.bottomWindow)
        
        
        mainlayout=QVBoxLayout(self)  
        mainlayout.addWidget(mainSplitter)
        self.setLayout(mainlayout)
        
#        self.resize(750,900) 
        
        QThread.sleep(1)
        
        


        
        
class LeftDialog(QDialog):  
    def __init__(self,parent=None):  
        super(LeftDialog,self).__init__(parent)  
#        font = QFont(self.tr("黑体"),12)
#        QApplication.setFont(font)
        
        labelCol=0  
        contentCol=1 
          
        label1=QLabel(self.tr("磁卡信息"))  
        label2=QLabel(self.tr("线路板尺寸："))  
        label3=QLabel(self.tr("H(高):"))  
        label4=QLabel(self.tr("W(宽):"))  
        plateHLineEdit=QLineEdit()  
        plateWLineEdit=QLineEdit() 
        leftTopLayout=QGridLayout()  
        leftTopLayout.addWidget(label1,0,labelCol)  
        leftTopLayout.addWidget(label2,1,labelCol)   
        leftTopLayout.addWidget(label3,2,labelCol)  
        leftTopLayout.addWidget(plateHLineEdit,2,contentCol)  
        leftTopLayout.addWidget(label4,3,labelCol)  
        leftTopLayout.addWidget(plateWLineEdit,3,contentCol)  
        
        
        label5=QLabel(self.tr("字符位置:"))  
        label6=QLabel(self.tr("X(LT):"))
        label7=QLabel(self.tr("Y(LT):"))
        label8=QLabel(self.tr("H(高):"))
        label9=QLabel(self.tr("W(宽):"))  
        charsXLineEdit=QLineEdit()  
        charsYLineEdit=QLineEdit()
        charsHLineEdit=QLineEdit()
        charsWLineEdit=QLineEdit()
        leftMiddleLayout=QGridLayout() 
        leftMiddleLayout.addWidget(label5,0,labelCol)  
        leftMiddleLayout.addWidget(label6,1,labelCol)  
        leftMiddleLayout.addWidget(charsXLineEdit,1,contentCol,1,2) 
        leftMiddleLayout.addWidget(label7,2,labelCol)  
        leftMiddleLayout.addWidget(charsYLineEdit,2,contentCol,1,2) 
        leftMiddleLayout.addWidget(label8,3,labelCol)  
        leftMiddleLayout.addWidget(charsHLineEdit,3,contentCol,1,2) 
        leftMiddleLayout.addWidget(label9,4,labelCol)  
        leftMiddleLayout.addWidget(charsWLineEdit,4,contentCol,1,2) 
        
        label10=QLabel(self.tr("目标字符:"))
        TarcharsLineEdit=QLineEdit() 
        leftBottomLayout=QGridLayout() 
        leftBottomLayout.addWidget(label10,0,labelCol)
        leftBottomLayout.addWidget(TarcharsLineEdit,0,contentCol,1,2)        
        leftBottomLayout.setColumnStretch(0,1)  
        leftBottomLayout.setColumnStretch(1,3)  
        
        mainLayout=QGridLayout(self)  
        mainLayout.setMargin(15)  
        mainLayout.setSpacing(10)  
        mainLayout.addLayout(leftTopLayout,0,0)   
        mainLayout.addLayout(leftMiddleLayout,1,0)  
        mainLayout.addLayout(leftBottomLayout,2,0)   
        mainLayout.setSizeConstraint(QLayout.SetFixedSize)  
  
    
class RightDialog(QDialog):  
    def __init__(self,inferenceChar,parent=None):  
        super(RightDialog,self).__init__(parent)
        self.inferenceChar = inferenceChar
#        font = QFont(self.tr("黑体"),12)
#        QApplication.setFont(font)
        
        labelCol=0  
        contentCol=1 
        
        label10 = QLabel(self.tr("识别信息"))
        label11=QLabel(self.tr("输入图片:"))  
        imgLabel=QLabel()  
        img=QPixmap("image/21.jpg")  
        imgLabel.setPixmap(img)  
        imgLabel.resize(img.width()*0.1,img.height()*0.1)  
        label12=QLabel(self.tr("提取字符区域:"))  
        charsLabel=QLabel()  
        chars=QPixmap("/home/zb/BoZhan/ocr_ws/test_img/1.bmp")  
        charsLabel.setPixmap(chars)  
        charsLabel.resize(chars.width(),chars.height()) 
        RightTopLayout=QVBoxLayout()  
        RightTopLayout.setSpacing(20) 
        RightTopLayout.addWidget(label10)
        RightTopLayout.addWidget(label11)  
        RightTopLayout.addWidget(imgLabel)  
        RightTopLayout.addWidget(label12)  
        RightTopLayout.addWidget(charsLabel) 
  
  
        label13=QLabel(self.tr("识别结果:")) 
        label14=QLabel(self.tr("识别用时:"))
        label15=QLabel(self.tr("是否匹配:"))
        self.charResLineEdit=QLineEdit()  
        self.timeCostLineEdit=QLineEdit()
        
        regOKLineEdit=QLineEdit("OK")       
        regOKLineEdit.setAutoFillBackground(True)
        pOK=regOKLineEdit.palette()
        colorOK = Qt.green
        pOK.setColor(QPalette.Base,colorOK)
        regOKLineEdit.setPalette(pOK)
        

        regNGLineEdit=QLineEdit("NG") 
        regNGLineEdit.setAutoFillBackground(True)
        pNG=regOKLineEdit.palette()
        colorNG = Qt.red
        pNG.setColor(QPalette.Base,colorNG)
        regNGLineEdit.setPalette(pNG)
        
        
        RightMiddleLayout=QGridLayout() 
        RightMiddleLayout.addWidget(label13,0,labelCol)
        RightMiddleLayout.addWidget(self.charResLineEdit,0,contentCol,1,2) 
        RightMiddleLayout.addWidget(label14,1,labelCol)  
        RightMiddleLayout.addWidget(self.timeCostLineEdit,1,contentCol,1,2) 
        RightMiddleLayout.addWidget(label15,2,labelCol) 
        RightMiddleLayout.addWidget(regOKLineEdit,2,contentCol,1,2) 
        RightMiddleLayout.addWidget(regNGLineEdit,2,contentCol+1,1,2) 
             
        label16=QLabel(self.tr("抓取位置:"))
        label17=QLabel(self.tr("X:"))
        label18=QLabel(self.tr("Y:"))
        pickXLineEdit=QLineEdit()  
        pickYLineEdit=QLineEdit() 
        RightBottomLayout=QGridLayout() 
        RightBottomLayout.addWidget(label16,0,labelCol)
        RightBottomLayout.addWidget(label17,1,labelCol)  
        RightBottomLayout.addWidget(pickXLineEdit,1,contentCol,1,2) 
        RightBottomLayout.addWidget(label18,2,labelCol) 
        RightBottomLayout.addWidget(pickYLineEdit,2,contentCol,1,2) 
        
        
        mainLayout=QGridLayout(self)  
        mainLayout.setMargin(15)  
        mainLayout.setSpacing(10)   
        mainLayout.addLayout(RightTopLayout,0,0)  
        mainLayout.addLayout(RightMiddleLayout,1,0) 
        mainLayout.addLayout(RightBottomLayout,2,0)  
        mainLayout.setSizeConstraint(QLayout.SetFixedSize) 
    
    def updateDisplay(self):       
        self.charResLineEdit.setText(QString(str(self.inferenceChar.digits_prediction_val)))
        self.timeCostLineEdit.setText(QString(str(self.inferenceChar.duration)))
        
    
        
        
class BottomDialog(QDialog): 
    update_chars = pyqtSignal()
    def __init__(self,inferenceChar,parent=None):  
        super(BottomDialog,self).__init__(parent) 
        self.inferenceChar = inferenceChar
#        font = QFont(self.tr("黑体"),12)
#        QApplication.setFont(font)
        
        labelCol=0  
        contentCol=1 

        
        self.startPushButton=QPushButton(self.tr("启动")) 
        self.stopPushButton=QPushButton(self.tr("暂停"))

        
        mainLayout=QGridLayout(self)  
        mainLayout.setMargin(15)  
        mainLayout.setSpacing(10)   
        mainLayout.addWidget(self.startPushButton,0,labelCol)  
        mainLayout.addWidget(self.stopPushButton,0,contentCol) 
        mainLayout.setSizeConstraint(QLayout.SetFixedSize) 
        
        self.connect(self.startPushButton, SIGNAL("clicked()"), self, SLOT("slotStart()"))
        self.connect(self.stopPushButton, SIGNAL("clicked()"), self, SLOT("slotStop()"))
        
#        startPushButton.clicked.connect(startRun())
#        stopPushButton.clicked.connect(stop())
        
    @pyqtSlot()    
    def slotStart(self):
        self.inferenceChar.startRun()
        self.update_chars.emit()

        
    @pyqtSlot()    
    def slotStop(self):
        self.inferenceChar.stop()


if __name__ == '__main__':
    import sys
    app=QApplication(sys.argv)  
    
    #start picture
    splash=QSplashScreen(QPixmap("image/23.png"))
    splash.show()  
    app.processEvents()
    
    CharInfer = InferenceMain()
    dialog=LayoutDialog(CharInfer) 
    dialog.bottomWindow.update_chars.connect(dialog.rightWindow.updateDisplay)
#    CharInfer.start()
    dialog.show()  
    app.exec_()  