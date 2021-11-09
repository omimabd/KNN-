# chargement du package datasets contenant plusieurs jeu de données
from sklearn.datasets import *
import pandas as pd  # Chargement de Pandas
import matplotlib.pyplot as plt  # import de Matplotlib
import numpy as np
# classe utilitaire pour découper les jeux de données
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # import de la classe de K-NN
# l'interface graphic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL.ImageQt import ImageQt

from matplotlib.figure import Figure
from PIL import Image


data = load_digits()  # chargement du dataset MNIST
data_dig = pd.DataFrame(data['data'][0:1797])  # Création d'un dataframe Panda
x_train = data.data  # les input variables
y_train = data.target  # les étiquettes (output variable)
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.25, shuffle=False)

# on veut entrainer un 7-NN Classifier (on utilise 7 voisins)
KNN = KNeighborsClassifier(5)


def apprentisage():
    KNN.fit(x_train, y_train)
    print("apprentissage réussi")


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 10, 421, 101))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(24)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(250, 140, 281, 271))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap(
            "../../../../../../../Photos-new-icon.png"))
        self.image.setObjectName("image")
        self.choisir = QtWidgets.QPushButton(self.centralwidget)
        self.choisir.setGeometry(QtCore.QRect(590, 250, 141, 41))
        self.choisir.setObjectName("choisir")
        self.resultat = QtWidgets.QLabel(self.centralwidget)
        self.resultat.setGeometry(QtCore.QRect(50, 460, 291, 51))
        self.resultat.setObjectName("resultat")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)

        self.label_2.setGeometry(QtCore.QRect(240, 460, 541, 51))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "KNN"))
        self.label.setText(_translate(
            "MainWindow", "Les k plus proches voisins"))
        self.choisir.setText(_translate("MainWindow", "Choisir une image"))
        self.resultat.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate(
            "MainWindow", ""))
        #self.pushButton.setText(_translate("MainWindow", "Test"))
        # self.pushButton.clicked.connect(self.test)

        self.choisir.clicked.connect(self.inputdial)
        # self.test.clicked.connect(self.test)

    def inputdial(self):

        self.roll, self.done = QtWidgets.QInputDialog.getInt(
            self.centralwidget, 'Image de test', 'choisissez de 1347:1797')
        fig = Figure(figsize=(5, 4), dpi=100)
        # self.displayImage(self.roll)
        sc = plt.imshow(data['images'][self.roll], cmap='Greys_r')
        image1 = Image.fromarray(np.uint8(sc.get_cmap()(sc.get_array())*255))
        if (self.done):
            qimage = ImageQt(image1)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            pixmap5 = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap5)
        # la précision par rapport aux données de test
        sc = KNN.score(x_test, y_test)
        print(sc)
        err = error = 1 - KNN.score(x_test, y_test)
        print('Erreur: %f' % err)
        # Afficher un élement de la matrice format image
        test = np.array(data['data'][self.roll])
        test1 = test.reshape(1, -1)
    # prédiction
        k = KNN.predict(test1)
        print(k)
        self.resultat.setText("le resultat est : "+str(k))
        self.label_2.setText("avec la probabilite suivante : " +
                             str(sc)+" et l'erreur suivant : "+str(err))


if __name__ == "__main__":
    import sys
    apprentisage()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
