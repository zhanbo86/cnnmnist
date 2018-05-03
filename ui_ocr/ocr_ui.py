# -*- coding: utf-8 -*-   
from PyQt4.QtGui import *  
from PyQt4.QtCore import *  
import sys  
from inferenceGUI import startRun,stop
  
QTextCodec.setCodecForTr(QTextCodec.codecForName("utf8"))  
  
class LayoutDialog(QDialog):  
    def __init__(self,parent=None):  
        super(LayoutDialog,self).__init__(parent)  
        self.setWindowTitle(self.tr("线路板批次OCR界面")) 
#        font = QFont(self.tr("黑体"),12)
#        QApplication.setFont(font)
        
        mainSplitter=QSplitter(Qt.Vertical,self)
        TopSplitter=QSplitter(Qt.Horizontal,mainSplitter)    
        TopSplitter.setOpaqueResize(False)
        
        leftFrame = QFrame(TopSplitter)
        rightFrame = QFrame(TopSplitter)
        
        leftWindow = LeftDialog()
        rightWindow = RightDialog()
        
        leftLayout = QGridLayout(leftFrame)
        leftLayout.addWidget(leftWindow)
        rightLayout = QGridLayout(rightFrame)
        rightLayout.addWidget(rightWindow)
        
        BottomFrame = QFrame(mainSplitter)
        bottomWindow = BottomDialog()
        bottomLayout = QGridLayout(BottomFrame)
        bottomLayout.addWidget(bottomWindow)
        
        
        mainlayout=QVBoxLayout(self)  
        mainlayout.addWidget(mainSplitter)
        self.setLayout(mainlayout)
        
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
    def __init__(self,parent=None):  
        super(RightDialog,self).__init__(parent)  
#        font = QFont(self.tr("黑体"),12)
#        QApplication.setFont(font)
        
        labelCol=0  
        contentCol=1 
        
        label10 = QLabel(self.tr("识别信息"))
        label11=QLabel(self.tr("输入图片:"))  
        imgLabel=QLabel()  
        img=QPixmap("image/2.jpg")  
        imgLabel.setPixmap(img)  
        imgLabel.resize(img.width()*0.1,img.height()*0.1)  
        label12=QLabel(self.tr("提取字符区域:"))  
        charsLabel=QLabel()  
        chars=QPixmap("image/3.jpg")  
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
        charResLineEdit=QLineEdit()  
        timeCostLineEdit=QLineEdit()
        
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
        RightMiddleLayout.addWidget(charResLineEdit,0,contentCol,1,2) 
        RightMiddleLayout.addWidget(label14,1,labelCol)  
        RightMiddleLayout.addWidget(timeCostLineEdit,1,contentCol,1,2) 
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
        
        
class BottomDialog(QDialog):  
    def __init__(self,parent=None):  
        super(BottomDialog,self).__init__(parent)  
#        font = QFont(self.tr("黑体"),12)
#        QApplication.setFont(font)
        
        labelCol=0  
        contentCol=1 

        
        startPushButton=QPushButton(self.tr("启动")) 
        stopPushButton=QPushButton(self.tr("暂停"))

        
        mainLayout=QGridLayout(self)  
        mainLayout.setMargin(15)  
        mainLayout.setSpacing(10)   
        mainLayout.addWidget(startPushButton,0,labelCol)  
        mainLayout.addWidget(stopPushButton,0,contentCol) 
        mainLayout.setSizeConstraint(QLayout.SetFixedSize) 
        
        self.startPushButton.clicked.connect(do_inference())
        self.stopPushButton.clicked.connect(stop())
   
app=QApplication(sys.argv)  

#start picture
splash=QSplashScreen(QPixmap("image/23.png"))
splash.show()  
app.processEvents()


dialog=LayoutDialog()  
dialog.show()  
app.exec_()  