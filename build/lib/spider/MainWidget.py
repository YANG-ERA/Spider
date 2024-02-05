
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, \
    QComboBox, QLabel, QSpinBox, QFileDialog

from paint3.PaintBoard import PaintBoard
import os

class MainWidget(QWidget):

    def __init__(self, Parent=None):

        super().__init__(Parent)

        self.__InitData()  # initial
        self.__InitView()

    def __InitData(self):

        self.__paintBoard = PaintBoard(self)
        self.__colorList = QColor.colorNames()

    def __InitView(self):

        self.setFixedSize(1200, 640)
        self.setWindowTitle("PaintBoard of PyQt5")

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)

        main_layout.addWidget(self.__paintBoard)

        sub_layout = QVBoxLayout()

        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("Clean")
        self.__btn_Clear.setParent(self)

        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("exit")
        self.__btn_Quit.setParent(self)
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("save picture")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__cbtn_Eraser = QCheckBox("use eraser")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        splitter = QSplitter(self)
        sub_layout.addWidget(splitter)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("Pen thickness")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(20)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(10)
        self.__spinBox_penThickness.setSingleStep(2)
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("Pen color")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)
        self.__comboBox_penColor.currentIndexChanged.connect(
            self.on_PenColorChange)
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        # savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        # print(savePath)
        # if savePath[0] == "":
        #     print("Save cancel")
        #     return
        #savePath = "C:\\Users\\quyang\\paint2\\test.png"
        ####
        image = self.__paintBoard.GetContentAsQImage()
        #image.save(savePath)

        image.save(os.getcwd()+"/test.jpg")

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True
        else:
            self.__paintBoard.EraserMode = False

    def Quit(self):
        self.close()
