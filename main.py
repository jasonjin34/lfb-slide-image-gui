import os
import os.path as ospath
import sys

import openslide as openslide
import glob
import openslide as osl
import numpy as np 
import cv2 as cv
from PIL.ImageQt import ImageQt

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *

from PatchExtractor import Extractor
from WorkerSignal import Worker, WorkerSignals, CursorSignals
from Viewer import Display, Scene

import time

os.environ.setdefault('PATH', '')

#get the absolute path for icons and pixmap image
#this will also be used in PyInstaller
def resource_path(relative_path):
    #get a temperator folder in the folder and get the folder get the base path
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    #combined the current path to get the absolute path
    return os.path.join(base_path, relative_path)


#define main windows
class Windows(QMainWindow):
    def __enter__(self):
        return self
    
    def __init__(self):
        QMainWindow.__init__(self)
        #setup the start path for Open file
        #init with home directory
        self.startpath = './GUI/'
        self.savedFilePath = None
        self.filedir = None 
        self.filedirNameList = [] #use for show all slide image in folders
        self.filedirFilePos = 0
        self.filename = None

        # ----------------------------------------------------------
        # initial setup: windows size, inital screen images etc
        # ----------------------------------------------------------
        self.setWindowTitle('HistoTool') 
        self.scene = Scene(parent=self)
        self.display = Display(self.scene, parent=self)
        self.display.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.display.setAlignment(QtCore.Qt.AlignCenter)
        self.display.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.display.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scene.setGraphicsView(self.display)
        self.setCentralWidget(self.display)
        self.resize(1600, 900)
        self.sidebarMessages = []
        self.display.show()
        self.display.setMouseTracking(True)

        #cursor effect
        self.scene.cursorSignal.press.connect(self.cursor_press)
        self.scene.cursorSignal.release.connect(self.cursor_release)

        # -----------------------------------------------------------
        # Toolbar
        # -----------------------------------------------------------
        toolbar = QToolBar(self)

        # -----------------------------------------------------------
        # Define All Buttons, and other Widgets 
        # add file buttons, save file buttons, undo, redo, zoomin, zoomout
        # -----------------------------------------------------------
        
        #setup the open slide file button 
        btnOpen = QPushButton(QIcon(resource_path('icons/blue-folder-horizontal-open.png')), "", self)
        btnOpen.setToolTip('(CTRL+O) open a slide')
        btnOpen.setShortcut('CTRL+O')
        btnOpen.clicked.connect(self.open)

        #setup the save file button
        btnSavefile = QPushButton(QIcon(resource_path('icons/blue-folder-import.png')), "", self)
        btnSavefile.setToolTip('(CTRL+S) select save file path')
        btnSavefile.setShortcut('CTRL+S')
        btnSavefile.clicked.connect(self.savefile)

        #setup the undo file button
        btnPrev = QPushButton(QIcon(resource_path('icons/undo.png')), "", self)
        btnPrev.setToolTip('(CTRL+L) previous slides image')
        btnPrev.setShortcut('CTRL+L')
        btnPrev.clicked.connect(self.prevImage)
        btnPrev.clicked.connect(self.view_slide)

        #setup the redo file button
        btnNext = QPushButton(QIcon(resource_path('icons/redo.png')), "", self)
        btnNext.setToolTip('(CTRL+R) next slides image')
        btnNext.setShortcut('CTRL+R')
        btnNext.clicked.connect(self.nextImage)
        btnNext.clicked.connect(self.view_slide)

        #setup the zoomin button
        btnZoomIn = QPushButton(QIcon(resource_path('icons/zoomin.png')), "", self)
        btnZoomIn.setToolTip('(CTRL+I) select save file path')
        btnZoomIn.setShortcut('CTRL+I')
        btnZoomIn.clicked.connect(self.zoom_in)

        #setup the view slide image
        btnViewSlide = QPushButton(QIcon(resource_path('icons/image-map.png')), "", self)
        btnViewSlide.setToolTip('(CTRL+V) view a slide image')
        btnViewSlide.setShortcut('CTRL+V')
        btnViewSlide.clicked.connect(self.view_slide)

        #setup the zommout button
        btnZoomOut = QPushButton(QIcon(resource_path('icons/zoomout.png')), "", self)
        btnZoomOut.setToolTip('(CTRL+O) select save file path')
        btnZoomOut.setShortcut('CTRL+O')
        btnZoomOut.clicked.connect(self.zoom_out)
        
        #setup open output folder
        btnReOpenfolder = QPushButton(QIcon(resource_path('icons/blue-folder-open.png')), "", self)
        btnReOpenfolder.setToolTip('re open the save folder')
        btnReOpenfolder.clicked.connect(self.reopen_output_folder)

        #setup information button
        btnInfo = QPushButton(QIcon(resource_path('icons/question.png')), "", self)
        btnInfo.setToolTip('Usage Information')
        btnInfo.clicked.connect(self.get_info_btn_func)

        #add all widgets to the toolbar
        toolbar.addWidget(btnOpen)
        toolbar.addWidget(btnSavefile)
        toolbar.addSeparator()

        toolbar.addWidget(btnViewSlide)
        toolbar.addWidget(btnPrev)
        toolbar.addWidget(btnNext)

        toolbar.addWidget(btnZoomIn)
        toolbar.addWidget(btnZoomOut)
        toolbar.addSeparator()
        
        toolbar.addWidget(btnInfo)
        
        toolbar.addWidget(btnReOpenfolder)        
        self.addToolBar(toolbar)
        

        # -----------------------------------------------------------
        # Sidebar
        # ----------------------------------------------------------
        self.sidebar = QDockWidget()

        #sidebar grid layerout
        self.sbGLWidget = QWidget(self.sidebar)
        self.sbGLWidget.setContentsMargins(0, 0, 0, 0)
        self.sbGLWidget.setGeometry(QRect(15, 10, 250, 800))
        self.sbGL = QGridLayout(self.sbGLWidget)
        self.sbGL.setContentsMargins(0, 0, 0, 0)

        self.sidebarLabelFilename = QLabel('no image loaded')
        self.sidebarLabelFilename.setAlignment(QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.sbGL.addWidget(self.sidebarLabelFilename, 0, 0, 1, 2)                                         

        #define button for patch extractor and data loader
        btnPatchExtractor = QPushButton('Patch \n Extractor')
        btnPatchExtractor.clicked.connect(self.patch_extractor_worker)
        btnPatchExtractor.setToolTip('Select files path and Save path first')

        btnDataLoader = QPushButton('Data \n Loader')
        btnDataLoader.clicked.connect(self.data_loader)

        #clear message button
        btnClearMsg = QPushButton('Clear')
        btnClearMsg.clicked.connect(self.clear_message)

        #define checkbox
        self.cbxmultithread = QCheckBox('Multithreading')
        self.cbxmultithread.toggled.connect(self.multithreadcbx)

        self.sbGL.addWidget(self.cbxmultithread, 2, 0, 1, 2)
        self.sbGL.addWidget(btnPatchExtractor, 3, 0)
        self.sbGL.addWidget(btnDataLoader, 3, 1)

        textLabel = QLabel('Processed Information')
        textLabel.setAlignment(QtCore.Qt.AlignBottom)
        self.sbGL.addWidget(textLabel, 4, 0, 1, 2)                                                    

        self.sidebarTextbox = QTextEdit()
        self.sidebarTextbox.setFontPointSize(9)
        self.sbGL.addWidget(self.sidebarTextbox, 5, 0, 1, 2)                                                 
        self.sbGL.addWidget(btnClearMsg, 6, 0, 1, 2)

        self.sbGL.setRowStretch(1, 2)
        self.sbGL.setRowStretch(2, 1)
        self.sbGL.setRowStretch(3, 1)
        self.sbGL.setRowStretch(4, 1)
        self.sbGL.setRowStretch(5, 8)
        self.sbGL.setRowStretch(6, 1)

        #setup sidebar with and location
        self.addDockWidget(Qt.RightDockWidgetArea, self.sidebar)
        self.sidebar.setMinimumWidth(285)
        self.sidebar.show()

        # ---------------------------------------------------------
        # IMPORTANT WITHOUT THIS FUNCTION MIGHT FREEZE
        # Setup Work class for multi thread processing
        # ---------------------------------------------------------
        self.threadpool = QThreadPool()
        threadinfostr = 'Multithreading with maximum {} threads'.format(self.threadpool.maxThreadCount())
        self.message(threadinfostr)
        self.finishstatus = False
        self.startstatus = False
        self.multithreadstatus = False

        # counting the execuate time
        self.timer = QElapsedTimer()
        self.timeconsumer = 0

        self.intervaltimer = QTimer()
        self.intervaltimer.setInterval(100)
        self.intervaltimer.timeout.connect(self.recurring_timer)
        self.intervaltimer.start()
        self.counter = 0

    #open the openslide file
    def open(self, filedir=None):
        if not filedir:
            filedir = QFileDialog.getExistingDirectory(self, 'Select OpenSlide Folder', self.startpath)
        self.filedir = str(filedir)
        self.sidebarLabelFilename.setText('image is loaded')

        #update the save files informations and setup in sidebar textbox
        filepathlistText = ['Slides name:',] 
        for name in glob.glob(filedir + '/*.*', recursive=True):
            filepathlistText.append(name.split('/')[-1])
            self.filedirNameList.append(name)
        tempstr = '\n -----'.join(filepathlistText)
        self.message('File folder: ' + filedir)
        self.message(tempstr)
        
        #inital filename and filename pos
        self.filename = self.filedirNameList[0]
    
    #view slide image
    def view_slide(self):
        if not self.filedir:
            return

        #return of getOpenFilename is tuple, first element is the address
        if self.filename is None:
            slidepath = QFileDialog.getOpenFileName(self, 'Select OpenSlide File', self.filedir)[0]
            self.filename = slidepath
            basefilename, filetype = ospath.splitext(ospath.basename(self.filename))        
            if filetype not in ['.SCN', '.svs', '.ndpi', '.tiff', '.svsslide', '.tif', '.bif', '.mrxs']:
                return
        
        #set scene and view
        self.scene.setNewImage(osl.OpenSlide(self.filename))
        self.display.update()
    
    #find the save file folder
    def savefile(self, filepath=None):
        if not filepath:
            self.savedFilePath = QFileDialog.getExistingDirectory(self, 'Select Save Path', self.startpath)
            self.message('Save Folder: ' + self.savedFilePath)

    #reopen the output folder
    def reopen_output_folder(self):
        #generate the error message
        if self.savedFilePath is None:
            errorMsg = QMessageBox()
            errorMsg.setIcon(QMessageBox.Warning)
            errorMsg.setText('Please select vaild save path first')
            errorMsg.setWindowTitle('ERROR')
            errorMsg.exec_()
        else:
            os.system('xdg-open ' + self.savedFilePath)
            
    def prevImage(self):
        tempIdx = self.filedirFilePos - 1
        if tempIdx < 0:
            self.filedirFilePos = 0
            return
        else:
            self.filedirFilePos = tempIdx
            self.filename = self.filedirNameList[self.filedirFilePos] 

    def nextImage(self):
        tempIdx = self.filedirFilePos + 1
        if tempIdx > len(self.filedirNameList) - 1:
            return
        else:
            self.filedirFilePos = tempIdx
            self.filename = self.filedirNameList[self.filedirFilePos] 
            self.view_slide()

    def zoom_in(self):
        self.scene.zoom_in()

    def zoom_out(self):
        self.scene.zoom_out()
    
    def cursor_press(self):
        self.setCursor(QCursor(Qt.SizeAllCursor))
    
    def cursor_release(self):
        self.setCursor(QCursor(Qt.ArrowCursor))

    def message(self, msg):
        self.sidebarMessages.append(msg)
        txt = '\n\n'.join(self.sidebarMessages)
        self.sidebarTextbox.setText(txt)

    def clear_message(self):
        self.sidebarMessages = []
        self.sidebarTextbox.setText('')
        self.savedFilePath = None
        self.filedir = None
        self.sidebarLabelFilename.setText('')
        self.startstatus = False
        self.finishstatus = False
    
    #image slide patch extractor for extractor one by one
    def patch_extractor(self):
        self.timer.start()
        if self.savedFilePath is None:
            outputdir = self.filedir + '_Patches'
        for file_in_path in glob.glob(self.filedir + '/*'):
            if file_in_path.endswith(('.ndpi', '.svs')):
                fileName, fileExtension = os.path.splitext(file_in_path)
                filePath , fileName = os.path.split(fileName)
                
                #avoid those files
                if '2017' or '2007' or '2027' or '2150' or '2179' or '2146' or '2175' in fileName[:4]:
                    pass                
                _, folderName = os.path.split(filePath)

                #get the extractor
                ex = Extractor(filePath=filePath, fileName=fileName, fileExtension=fileExtension, kidneyIdentifier=1) 
                outputPath = os.path.join(self.savedFilePath, folderName, fileName)
                if not os.path.exists(outputPath):
                    os.makedirs(outputPath, exist_ok=True)
                ex.get_only_patches(withinKidney=True, outputPath=outputPath)
        self.timeconsumer = self.timer.elapsed() / 1000
    
    # extractor patcher simutaneously
    def patch_extractor_list(self):
        self.timer.start()
        extractorList = []
        outputpathList = []

        if self.savedFilePath is None:
            outputdir = self.filedir + '_Patches'
        for file_in_path in glob.glob(self.filedir + '/*'):
            if file_in_path.endswith(('.ndpi', '.svs')):
                fileName, fileExtension = os.path.splitext(file_in_path)
                filePath , fileName = os.path.split(fileName)
                
                #avoid those files:
                if '2017' or '2007' or '2027' or '2150' or '2179' or '2146' or '2175' in fileName[:4]:
                    pass                
                _, folderName = os.path.split(filePath)

                #get the extractor
                ex = Extractor(filePath=filePath, fileName=fileName, fileExtension=fileExtension, kidneyIdentifier=1) 
                outputPath = os.path.join(self.savedFilePath, folderName, fileName)
                if not os.path.exists(outputPath):
                    os.makedirs(outputPath, exist_ok=True)
                extractorList.append(ex)
                outputpathList.append(outputPath)
        return extractorList, outputpathList
    
    def extract_exec(self, ex, path):
        ex.get_only_patches(withinKidney=False, outputPath=path)
        self.timeconsumer = self.timer.elapsed() / 1000
    
    def data_loader(self):
        pass
    
    # ---------------------------------------------------------
    # thread help functions, thread worker multithreading wait for all threads to finish pyqtstart, finish
    # processing
    # ---------------------------------------------------------
    def multithreadcbx(self): # multithreading checkbox
        status = self.cbxmultithread.isChecked()
        if status:
            self.message('Select Multithreading Processing')
            self.multithreadstatus = True
        else:
            self.multithreadstatus = False

    def thread_finish(self, id):
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.finishstatus = True
        self.message('Processing time of Thread {}: {} secs'.format(id, self.timer.elapsed() / 1000))
    
    def thread_start(self, num):
        self.setCursor(QCursor(Qt.WaitCursor))
        self.counter = 0
        self.startstatus = True
        msg = 'Start Process {}'.format(num)
        self.message(msg)
    
    def recurring_timer(self):
        msg = 25 *'=='
        pos = self.counter % 25
        even = (self.counter // 25) % 2
        if self.startstatus and not self.finishstatus:
            if not even:
                tmpmsg = msg[:pos]
            else:
                tmpmsg = msg[: 25 -pos]
            self.sidebarLabelFilename.setText('Processing ' + tmpmsg)
        elif self.startstatus and self.finishstatus:
            self.sidebarLabelFilename.setText('Process Finish')
        self.counter += 1 

    # ----------------------------------------------------------------------
    # multi thread function implementation
    # ----------------------------------------------------------------------
    def multi_thread_test(self):
        self.timer.start()
        for x in range(5):
            time.sleep(1)

    def patch_extractor_worker(self):
        self.startstatus = True
        self.finishstatus = False
        nonparalFn = self.patch_extractor # non parallel function
        if not self.savedFilePath:
            return
        if not self.multithreadstatus:
            worker = Worker(nonparalFn)
            worker.signals.started.connect(lambda x=1: self.thread_start(x))
            worker.signals.finished.connect(lambda x=1: self.thread_finish(x))
            self.threadpool.start(worker)
        else:
            extractorList, pathList = self.patch_extractor_list()
            threadid = [id+1 for id in range(len(pathList))]
            for ex, path, id in zip(extractorList, pathList, threadid):
                worker = Worker(self.extract_exec, ex, path)
                worker.signals.started.connect(lambda x=id: self.thread_start(x))
                worker.signals.finished.connect(lambda x=id: self.thread_finish(x))
                self.threadpool.start(worker)
    
    def get_info_btn_func(self):
        infoMsg = QMessageBox()
        infoMsg.setIcon(QMessageBox.Question)
        infoMsg.setText('Essential Information: \n Patch extractor: select the file directory and output path.\
                         \n Slide Image review: select the slides image directory')
        infoMsg.setWindowTitle('User Guide')
        infoMsg.exec_()

#execution the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # initial GUI window
    win = Windows() 
    win.__init__()
    win.show()
    sys.exit(app.exec_())
