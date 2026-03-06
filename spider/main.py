from PyQt5.QtWidgets import QApplication
import sys


from paint2.MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication

import sys


def main():
    app = QApplication(sys.argv)

    mainWidget = MainWidget()
    mainWidget.show()

    exit(app.exec_())


if __name__ == '__main__':
    main()
