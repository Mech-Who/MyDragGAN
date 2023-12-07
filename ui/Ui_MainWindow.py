# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
    QGridLayout, QLabel, QLineEdit, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QSpacerItem,
    QStatusBar, QToolButton, QWidget)

from components.ImageWidget import ImageWidget

class Ui_DragGAN(object):
    def setupUi(self, DragGAN):
        if not DragGAN.objectName():
            DragGAN.setObjectName(u"DragGAN")
        DragGAN.resize(1076, 834)
        self.centralwidget = QWidget(DragGAN)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName(u"gridLayout")
        self.experiment = QWidget(self.centralwidget)
        self.experiment.setObjectName(u"experiment")
        self.gridLayout_5 = QGridLayout(self.experiment)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.OnePoint_CheckBox = QCheckBox(self.experiment)
        self.Point_ButtonGroup = QButtonGroup(DragGAN)
        self.Point_ButtonGroup.setObjectName(u"Point_ButtonGroup")
        self.Point_ButtonGroup.addButton(self.OnePoint_CheckBox)
        self.OnePoint_CheckBox.setObjectName(u"OnePoint_CheckBox")
        self.OnePoint_CheckBox.setEnabled(True)
        self.OnePoint_CheckBox.setChecked(False)

        self.gridLayout_5.addWidget(self.OnePoint_CheckBox, 3, 1, 1, 1)

        self.SevenBlock_CheckBox = QCheckBox(self.experiment)
        self.Block_ButtonGroup = QButtonGroup(DragGAN)
        self.Block_ButtonGroup.setObjectName(u"Block_ButtonGroup")
        self.Block_ButtonGroup.setExclusive(True)
        self.Block_ButtonGroup.addButton(self.SevenBlock_CheckBox)
        self.SevenBlock_CheckBox.setObjectName(u"SevenBlock_CheckBox")
        self.SevenBlock_CheckBox.setEnabled(True)
        self.SevenBlock_CheckBox.setChecked(False)

        self.gridLayout_5.addWidget(self.SevenBlock_CheckBox, 4, 4, 1, 1)

        self.FourBlock_CheckBox = QCheckBox(self.experiment)
        self.Block_ButtonGroup.addButton(self.FourBlock_CheckBox)
        self.FourBlock_CheckBox.setObjectName(u"FourBlock_CheckBox")
        self.FourBlock_CheckBox.setEnabled(True)
        self.FourBlock_CheckBox.setChecked(False)

        self.gridLayout_5.addWidget(self.FourBlock_CheckBox, 4, 1, 1, 1)

        self.SaveExperiment_PushButton = QPushButton(self.experiment)
        self.SaveExperiment_PushButton.setObjectName(u"SaveExperiment_PushButton")

        self.gridLayout_5.addWidget(self.SaveExperiment_PushButton, 0, 4, 1, 1)

        self.TargetImage_Label = QLabel(self.experiment)
        self.TargetImage_Label.setObjectName(u"TargetImage_Label")

        self.gridLayout_5.addWidget(self.TargetImage_Label, 0, 1, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_5.addItem(self.verticalSpacer_4, 5, 1, 1, 1)

        self.TargetImage_LineEdit = QLineEdit(self.experiment)
        self.TargetImage_LineEdit.setObjectName(u"TargetImage_LineEdit")
        self.TargetImage_LineEdit.setEnabled(False)

        self.gridLayout_5.addWidget(self.TargetImage_LineEdit, 2, 1, 1, 3)

        self.Experiment_Label = QLabel(self.experiment)
        self.Experiment_Label.setObjectName(u"Experiment_Label")

        self.gridLayout_5.addWidget(self.Experiment_Label, 0, 0, 1, 1)

        self.SixBlock_CheckBox = QCheckBox(self.experiment)
        self.Block_ButtonGroup.addButton(self.SixBlock_CheckBox)
        self.SixBlock_CheckBox.setObjectName(u"SixBlock_CheckBox")
        self.SixBlock_CheckBox.setEnabled(True)
        self.SixBlock_CheckBox.setChecked(True)

        self.gridLayout_5.addWidget(self.SixBlock_CheckBox, 4, 3, 1, 1)

        self.TargetImage_ToolButton = QToolButton(self.experiment)
        self.TargetImage_ToolButton.setObjectName(u"TargetImage_ToolButton")

        self.gridLayout_5.addWidget(self.TargetImage_ToolButton, 2, 4, 1, 1)

        self.LatentBlock_Label = QLabel(self.experiment)
        self.LatentBlock_Label.setObjectName(u"LatentBlock_Label")

        self.gridLayout_5.addWidget(self.LatentBlock_Label, 4, 0, 1, 1)

        self.Test_PushButton = QPushButton(self.experiment)
        self.Test_PushButton.setObjectName(u"Test_PushButton")

        self.gridLayout_5.addWidget(self.Test_PushButton, 0, 3, 1, 1)

        self.FiveBlock_CheckBox = QCheckBox(self.experiment)
        self.Block_ButtonGroup.addButton(self.FiveBlock_CheckBox)
        self.FiveBlock_CheckBox.setObjectName(u"FiveBlock_CheckBox")
        self.FiveBlock_CheckBox.setEnabled(True)
        self.FiveBlock_CheckBox.setChecked(False)

        self.gridLayout_5.addWidget(self.FiveBlock_CheckBox, 4, 2, 1, 1)

        self.FivePoints_CheckBox = QCheckBox(self.experiment)
        self.Point_ButtonGroup.addButton(self.FivePoints_CheckBox)
        self.FivePoints_CheckBox.setObjectName(u"FivePoints_CheckBox")

        self.gridLayout_5.addWidget(self.FivePoints_CheckBox, 3, 2, 1, 1)

        self.SixtyEightPoints_CheckBox = QCheckBox(self.experiment)
        self.Point_ButtonGroup.addButton(self.SixtyEightPoints_CheckBox)
        self.SixtyEightPoints_CheckBox.setObjectName(u"SixtyEightPoints_CheckBox")
        self.SixtyEightPoints_CheckBox.setChecked(True)

        self.gridLayout_5.addWidget(self.SixtyEightPoints_CheckBox, 3, 3, 1, 1)


        self.gridLayout.addWidget(self.experiment, 2, 0, 1, 1)

        self.model = QWidget(self.centralwidget)
        self.model.setObjectName(u"model")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(14)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.model.sizePolicy().hasHeightForWidth())
        self.model.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QGridLayout(self.model)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.Browse_PushButton = QPushButton(self.model)
        self.Browse_PushButton.setObjectName(u"Browse_PushButton")

        self.gridLayout_2.addWidget(self.Browse_PushButton, 2, 3, 1, 4)

        self.Minus4Seed_PushButton = QPushButton(self.model)
        self.Minus4Seed_PushButton.setObjectName(u"Minus4Seed_PushButton")
        self.Minus4Seed_PushButton.setMinimumSize(QSize(75, 0))

        self.gridLayout_2.addWidget(self.Minus4Seed_PushButton, 3, 2, 1, 1)

        self.Recent_PushButton = QPushButton(self.model)
        self.Recent_PushButton.setObjectName(u"Recent_PushButton")

        self.gridLayout_2.addWidget(self.Recent_PushButton, 2, 1, 1, 2)

        self.Wp_CheckBox = QCheckBox(self.model)
        self.LatentSpace_ButtonGroup = QButtonGroup(DragGAN)
        self.LatentSpace_ButtonGroup.setObjectName(u"LatentSpace_ButtonGroup")
        self.LatentSpace_ButtonGroup.addButton(self.Wp_CheckBox)
        self.Wp_CheckBox.setObjectName(u"Wp_CheckBox")

        self.gridLayout_2.addWidget(self.Wp_CheckBox, 5, 2, 1, 1)

        self.Device_ComboBox = QComboBox(self.model)
        self.Device_ComboBox.setObjectName(u"Device_ComboBox")

        self.gridLayout_2.addWidget(self.Device_ComboBox, 0, 1, 1, 6)

        self.W_CheckBox = QCheckBox(self.model)
        self.LatentSpace_ButtonGroup.addButton(self.W_CheckBox)
        self.W_CheckBox.setObjectName(u"W_CheckBox")
        self.W_CheckBox.setEnabled(True)
        self.W_CheckBox.setChecked(True)

        self.gridLayout_2.addWidget(self.W_CheckBox, 5, 1, 1, 1)

        self.Generate_PushButton = QPushButton(self.model)
        self.Generate_PushButton.setObjectName(u"Generate_PushButton")

        self.gridLayout_2.addWidget(self.Generate_PushButton, 6, 1, 1, 1)

        self.Seed_LineEdit = QLineEdit(self.model)
        self.Seed_LineEdit.setObjectName(u"Seed_LineEdit")
        self.Seed_LineEdit.setEnabled(True)

        self.gridLayout_2.addWidget(self.Seed_LineEdit, 3, 1, 1, 1)

        self.Pickle_LineEdit = QLineEdit(self.model)
        self.Pickle_LineEdit.setObjectName(u"Pickle_LineEdit")
        self.Pickle_LineEdit.setEnabled(False)

        self.gridLayout_2.addWidget(self.Pickle_LineEdit, 1, 1, 1, 6)

        self.Device_Label = QLabel(self.model)
        self.Device_Label.setObjectName(u"Device_Label")

        self.gridLayout_2.addWidget(self.Device_Label, 0, 0, 1, 1)

        self.Latent_Label = QLabel(self.model)
        self.Latent_Label.setObjectName(u"Latent_Label")

        self.gridLayout_2.addWidget(self.Latent_Label, 3, 0, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_3, 7, 2, 1, 1)

        self.Plus4Seed_PushButton = QPushButton(self.model)
        self.Plus4Seed_PushButton.setObjectName(u"Plus4Seed_PushButton")
        self.Plus4Seed_PushButton.setMinimumSize(QSize(75, 0))

        self.gridLayout_2.addWidget(self.Plus4Seed_PushButton, 3, 3, 1, 1)

        self.Pickle_Label = QLabel(self.model)
        self.Pickle_Label.setObjectName(u"Pickle_Label")

        self.gridLayout_2.addWidget(self.Pickle_Label, 1, 0, 1, 1)

        self.Seed_Label = QLabel(self.model)
        self.Seed_Label.setObjectName(u"Seed_Label")

        self.gridLayout_2.addWidget(self.Seed_Label, 3, 4, 1, 1)

        self.RandomSeed_CheckBox = QCheckBox(self.model)
        self.RandomSeed_CheckBox.setObjectName(u"RandomSeed_CheckBox")
        self.RandomSeed_CheckBox.setEnabled(True)
        self.RandomSeed_CheckBox.setChecked(False)

        self.gridLayout_2.addWidget(self.RandomSeed_CheckBox, 4, 1, 1, 1)

        self.SaveReal_PushButton = QPushButton(self.model)
        self.SaveReal_PushButton.setObjectName(u"SaveReal_PushButton")

        self.gridLayout_2.addWidget(self.SaveReal_PushButton, 6, 2, 1, 1)

        self.gridLayout_2.setRowStretch(0, 1)
        self.gridLayout_2.setColumnStretch(0, 2)

        self.gridLayout.addWidget(self.model, 0, 0, 1, 1)

        self.mask = QWidget(self.centralwidget)
        self.mask.setObjectName(u"mask")
        self.gridLayout_4 = QGridLayout(self.mask)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.Plus4Radius_PushButton = QPushButton(self.mask)
        self.Plus4Radius_PushButton.setObjectName(u"Plus4Radius_PushButton")

        self.gridLayout_4.addWidget(self.Plus4Radius_PushButton, 2, 3, 1, 1)

        self.Minus4Lambda_PushButton = QPushButton(self.mask)
        self.Minus4Lambda_PushButton.setObjectName(u"Minus4Lambda_PushButton")

        self.gridLayout_4.addWidget(self.Minus4Lambda_PushButton, 3, 2, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_4.addItem(self.verticalSpacer_2, 4, 2, 1, 1)

        self.Lambda_LineEdit = QLineEdit(self.mask)
        self.Lambda_LineEdit.setObjectName(u"Lambda_LineEdit")
        self.Lambda_LineEdit.setEnabled(False)

        self.gridLayout_4.addWidget(self.Lambda_LineEdit, 3, 1, 1, 1)

        self.ShowMask_CheckBox = QCheckBox(self.mask)
        self.ShowMask_CheckBox.setObjectName(u"ShowMask_CheckBox")

        self.gridLayout_4.addWidget(self.ShowMask_CheckBox, 1, 3, 1, 2)

        self.ResetMask_PushButton = QPushButton(self.mask)
        self.ResetMask_PushButton.setObjectName(u"ResetMask_PushButton")

        self.gridLayout_4.addWidget(self.ResetMask_PushButton, 1, 1, 1, 2)

        self.Mask_Label = QLabel(self.mask)
        self.Mask_Label.setObjectName(u"Mask_Label")

        self.gridLayout_4.addWidget(self.Mask_Label, 0, 0, 1, 1)

        self.Lambda_Label = QLabel(self.mask)
        self.Lambda_Label.setObjectName(u"Lambda_Label")

        self.gridLayout_4.addWidget(self.Lambda_Label, 3, 4, 1, 1)

        self.Radius_Label = QLabel(self.mask)
        self.Radius_Label.setObjectName(u"Radius_Label")

        self.gridLayout_4.addWidget(self.Radius_Label, 2, 4, 1, 1)

        self.Radius_LineEdit = QLineEdit(self.mask)
        self.Radius_LineEdit.setObjectName(u"Radius_LineEdit")
        self.Radius_LineEdit.setEnabled(False)

        self.gridLayout_4.addWidget(self.Radius_LineEdit, 2, 1, 1, 1)

        self.FlexibleArea_PushButton = QPushButton(self.mask)
        self.FlexibleArea_PushButton.setObjectName(u"FlexibleArea_PushButton")

        self.gridLayout_4.addWidget(self.FlexibleArea_PushButton, 0, 1, 1, 2)

        self.Minus4Radius_PushButton = QPushButton(self.mask)
        self.Minus4Radius_PushButton.setObjectName(u"Minus4Radius_PushButton")

        self.gridLayout_4.addWidget(self.Minus4Radius_PushButton, 2, 2, 1, 1)

        self.Plus4Lambda_PushButton = QPushButton(self.mask)
        self.Plus4Lambda_PushButton.setObjectName(u"Plus4Lambda_PushButton")

        self.gridLayout_4.addWidget(self.Plus4Lambda_PushButton, 3, 3, 1, 1)

        self.FixedArea_PushButton = QPushButton(self.mask)
        self.FixedArea_PushButton.setObjectName(u"FixedArea_PushButton")

        self.gridLayout_4.addWidget(self.FixedArea_PushButton, 0, 3, 1, 2)


        self.gridLayout.addWidget(self.mask, 4, 0, 1, 1)

        self.Image_Widget = ImageWidget(self.centralwidget)
        self.Image_Widget.setObjectName(u"Image_Widget")

        self.gridLayout.addWidget(self.Image_Widget, 0, 1, 5, 1)

        self.drag = QWidget(self.centralwidget)
        self.drag.setObjectName(u"drag")
        self.gridLayout_3 = QGridLayout(self.drag)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.R1_LineEdit = QLineEdit(self.drag)
        self.R1_LineEdit.setObjectName(u"R1_LineEdit")

        self.gridLayout_3.addWidget(self.R1_LineEdit, 5, 1, 1, 2)

        self.StepNumber_Label = QLabel(self.drag)
        self.StepNumber_Label.setObjectName(u"StepNumber_Label")

        self.gridLayout_3.addWidget(self.StepNumber_Label, 8, 2, 1, 1)

        self.Stop_PushButton = QPushButton(self.drag)
        self.Stop_PushButton.setObjectName(u"Stop_PushButton")

        self.gridLayout_3.addWidget(self.Stop_PushButton, 1, 4, 1, 1)

        self.Reset4R2_PushButton = QPushButton(self.drag)
        self.Reset4R2_PushButton.setObjectName(u"Reset4R2_PushButton")

        self.gridLayout_3.addWidget(self.Reset4R2_PushButton, 6, 4, 1, 1)

        self.StepSize_Label = QLabel(self.drag)
        self.StepSize_Label.setObjectName(u"StepSize_Label")

        self.gridLayout_3.addWidget(self.StepSize_Label, 4, 3, 1, 1)

        self.StepSize_LineEdit = QLineEdit(self.drag)
        self.StepSize_LineEdit.setObjectName(u"StepSize_LineEdit")

        self.gridLayout_3.addWidget(self.StepSize_LineEdit, 4, 1, 1, 2)

        self.R2_LineEdit = QLineEdit(self.drag)
        self.R2_LineEdit.setObjectName(u"R2_LineEdit")

        self.gridLayout_3.addWidget(self.R2_LineEdit, 6, 1, 1, 2)

        self.SaveGenerate_PushButton = QPushButton(self.drag)
        self.SaveGenerate_PushButton.setObjectName(u"SaveGenerate_PushButton")

        self.gridLayout_3.addWidget(self.SaveGenerate_PushButton, 8, 3, 1, 1)

        self.Start_PushButton = QPushButton(self.drag)
        self.Start_PushButton.setObjectName(u"Start_PushButton")

        self.gridLayout_3.addWidget(self.Start_PushButton, 1, 1, 1, 3)

        self.ResetPoint_PushButton = QPushButton(self.drag)
        self.ResetPoint_PushButton.setObjectName(u"ResetPoint_PushButton")

        self.gridLayout_3.addWidget(self.ResetPoint_PushButton, 0, 4, 1, 1)

        self.Reset4R1_PushButton = QPushButton(self.drag)
        self.Reset4R1_PushButton.setObjectName(u"Reset4R1_PushButton")

        self.gridLayout_3.addWidget(self.Reset4R1_PushButton, 5, 4, 1, 1)

        self.Reset4StepSize_PushButton = QPushButton(self.drag)
        self.Reset4StepSize_PushButton.setObjectName(u"Reset4StepSize_PushButton")

        self.gridLayout_3.addWidget(self.Reset4StepSize_PushButton, 4, 4, 1, 1)

        self.AddPoint_PushButton = QPushButton(self.drag)
        self.AddPoint_PushButton.setObjectName(u"AddPoint_PushButton")

        self.gridLayout_3.addWidget(self.AddPoint_PushButton, 0, 1, 1, 3)

        self.Drag_Label = QLabel(self.drag)
        self.Drag_Label.setObjectName(u"Drag_Label")

        self.gridLayout_3.addWidget(self.Drag_Label, 0, 0, 1, 1)

        self.Steps_Label = QLabel(self.drag)
        self.Steps_Label.setObjectName(u"Steps_Label")

        self.gridLayout_3.addWidget(self.Steps_Label, 8, 1, 1, 1)

        self.R1_Label = QLabel(self.drag)
        self.R1_Label.setObjectName(u"R1_Label")

        self.gridLayout_3.addWidget(self.R1_Label, 5, 3, 1, 1)

        self.R2_Label = QLabel(self.drag)
        self.R2_Label.setObjectName(u"R2_Label")

        self.gridLayout_3.addWidget(self.R2_Label, 6, 3, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer, 9, 2, 1, 1)

        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_3.setColumnStretch(2, 1)
        self.gridLayout_3.setColumnStretch(3, 2)
        self.gridLayout_3.setColumnStretch(4, 2)

        self.gridLayout.addWidget(self.drag, 1, 0, 1, 1)

        self.gridLayout.setRowStretch(0, 7)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 3)
        DragGAN.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(DragGAN)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1076, 21))
        DragGAN.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(DragGAN)
        self.statusbar.setObjectName(u"statusbar")
        DragGAN.setStatusBar(self.statusbar)

        self.retranslateUi(DragGAN)

        QMetaObject.connectSlotsByName(DragGAN)
    # setupUi

    def retranslateUi(self, DragGAN):
        DragGAN.setWindowTitle(QCoreApplication.translate("DragGAN", u"MainWindow", None))
        self.OnePoint_CheckBox.setText(QCoreApplication.translate("DragGAN", u"1 point", None))
        self.SevenBlock_CheckBox.setText(QCoreApplication.translate("DragGAN", u"7", None))
        self.FourBlock_CheckBox.setText(QCoreApplication.translate("DragGAN", u"4", None))
        self.SaveExperiment_PushButton.setText(QCoreApplication.translate("DragGAN", u"Save", None))
        self.TargetImage_Label.setText(QCoreApplication.translate("DragGAN", u"Target Image", None))
        self.Experiment_Label.setText(QCoreApplication.translate("DragGAN", u"Experiment", None))
        self.SixBlock_CheckBox.setText(QCoreApplication.translate("DragGAN", u"6", None))
        self.TargetImage_ToolButton.setText(QCoreApplication.translate("DragGAN", u"...", None))
        self.LatentBlock_Label.setText(QCoreApplication.translate("DragGAN", u"Latent Block", None))
        self.Test_PushButton.setText(QCoreApplication.translate("DragGAN", u"Test", None))
        self.FiveBlock_CheckBox.setText(QCoreApplication.translate("DragGAN", u"5", None))
        self.FivePoints_CheckBox.setText(QCoreApplication.translate("DragGAN", u"5 points", None))
        self.SixtyEightPoints_CheckBox.setText(QCoreApplication.translate("DragGAN", u"68 points", None))
        self.Browse_PushButton.setText(QCoreApplication.translate("DragGAN", u"Browse...", None))
        self.Minus4Seed_PushButton.setText(QCoreApplication.translate("DragGAN", u"-", None))
        self.Recent_PushButton.setText(QCoreApplication.translate("DragGAN", u"Recent...", None))
        self.Wp_CheckBox.setText(QCoreApplication.translate("DragGAN", u"W+", None))
        self.W_CheckBox.setText(QCoreApplication.translate("DragGAN", u"W", None))
        self.Generate_PushButton.setText(QCoreApplication.translate("DragGAN", u"Generate", None))
        self.Seed_LineEdit.setText(QCoreApplication.translate("DragGAN", u"0", None))
        self.Device_Label.setText(QCoreApplication.translate("DragGAN", u"Device", None))
        self.Latent_Label.setText(QCoreApplication.translate("DragGAN", u"Latent", None))
        self.Plus4Seed_PushButton.setText(QCoreApplication.translate("DragGAN", u"+", None))
        self.Pickle_Label.setText(QCoreApplication.translate("DragGAN", u"Pickle", None))
        self.Seed_Label.setText(QCoreApplication.translate("DragGAN", u"Seed", None))
        self.RandomSeed_CheckBox.setText(QCoreApplication.translate("DragGAN", u"Random Seed", None))
        self.SaveReal_PushButton.setText(QCoreApplication.translate("DragGAN", u"Save", None))
        self.Plus4Radius_PushButton.setText(QCoreApplication.translate("DragGAN", u"+", None))
        self.Minus4Lambda_PushButton.setText(QCoreApplication.translate("DragGAN", u"-", None))
        self.Lambda_LineEdit.setText(QCoreApplication.translate("DragGAN", u"0.5", None))
        self.ShowMask_CheckBox.setText(QCoreApplication.translate("DragGAN", u"Show mask", None))
        self.ResetMask_PushButton.setText(QCoreApplication.translate("DragGAN", u"Reset mask", None))
        self.Mask_Label.setText(QCoreApplication.translate("DragGAN", u"Mask", None))
        self.Lambda_Label.setText(QCoreApplication.translate("DragGAN", u"Lambda", None))
        self.Radius_Label.setText(QCoreApplication.translate("DragGAN", u"Radius", None))
        self.Radius_LineEdit.setText(QCoreApplication.translate("DragGAN", u"1", None))
        self.FlexibleArea_PushButton.setText(QCoreApplication.translate("DragGAN", u"Flexible area", None))
        self.Minus4Radius_PushButton.setText(QCoreApplication.translate("DragGAN", u"-", None))
        self.Plus4Lambda_PushButton.setText(QCoreApplication.translate("DragGAN", u"+", None))
        self.FixedArea_PushButton.setText(QCoreApplication.translate("DragGAN", u"Fixed area", None))
        self.StepNumber_Label.setText(QCoreApplication.translate("DragGAN", u"0", None))
        self.Stop_PushButton.setText(QCoreApplication.translate("DragGAN", u"Stop", None))
        self.Reset4R2_PushButton.setText(QCoreApplication.translate("DragGAN", u"Reset", None))
        self.StepSize_Label.setText(QCoreApplication.translate("DragGAN", u"Step Size", None))
        self.SaveGenerate_PushButton.setText(QCoreApplication.translate("DragGAN", u"Save", None))
        self.Start_PushButton.setText(QCoreApplication.translate("DragGAN", u"Start", None))
        self.ResetPoint_PushButton.setText(QCoreApplication.translate("DragGAN", u"Reset point", None))
        self.Reset4R1_PushButton.setText(QCoreApplication.translate("DragGAN", u"Reset", None))
        self.Reset4StepSize_PushButton.setText(QCoreApplication.translate("DragGAN", u"Reset", None))
        self.AddPoint_PushButton.setText(QCoreApplication.translate("DragGAN", u"Add point", None))
        self.Drag_Label.setText(QCoreApplication.translate("DragGAN", u"Drag", None))
        self.Steps_Label.setText(QCoreApplication.translate("DragGAN", u"Steps:", None))
        self.R1_Label.setText(QCoreApplication.translate("DragGAN", u"R1", None))
        self.R2_Label.setText(QCoreApplication.translate("DragGAN", u"R2", None))
    # retranslateUi

