#https://blog.csdn.net/CreatorGG/article/details/81542837?spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-81542837-blog-118923269.pc_relevant_multi_platform_featuressortv2dupreplace&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-81542837-blog-118923269.pc_relevant_multi_platform_featuressortv2dupreplace&utm_relevant_index=20
'''
Created on 2018年8月9日

@author: Freedom
'''
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, \
    QColor, QSize
from PyQt5.QtCore import Qt


class PaintBoard(QWidget):

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData()
        self.__InitView()

    def __InitData(self):

        self.__size = QSize(1000, 600)

        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.white)

        self.__IsEmpty = True
        self.EraserMode = False

        self.__lastPos = QPoint(0, 0)
        self.__currentPos = QPoint(0, 0)

        self.__painter = QPainter()

        self.__thickness = 10
        self.__penColor = QColor("black")
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        self.setFixedSize(self.__size)

    def Clear(self):
        self.__board.fill(Qt.white)
        self.update()
        self.__IsEmpty = True

    def ChangePenColor(self, color="black"):
        self.__penColor = QColor(color)

    def ChangePenThickness(self, thickness=10):
        self.__thickness = thickness

    def IsEmpty(self):
        return self.__IsEmpty

    def GetContentAsQImage(self):
        image = self.__board.toImage()
        return image

    def paintEvent(self, paintEvent):
        self.__painter.begin(self)
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.__board)

        if self.EraserMode == False:
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))  # 设置画笔颜色，粗细
        else:
            self.__painter.setPen(QPen(Qt.white, 10))

        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False 



