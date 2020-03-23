import os 
import sys

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *

from PIL.ImageQt import ImageQt
from WorkerSignal import CursorSignals

os.environ.setdefault('PATH', '')

#get the absolute path for icons and pixmap image
#this will also be used in PyInstaller
def resource_path(relative_path):
    #get a temperator folder in the folder and get the folder get the base path
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    #combined the current path to get the absolute path
    return os.path.join(base_path, relative_path)

#define graphc view
class Display(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent=parent)

#define scene, for the future usage, like display the slide images, etc
class Scene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.slide = None
        self.slidePixmap = self.addPixmap(QPixmap.fromImage(QImage(resource_path('icons/startscreen.png'))))
        self.display = None

        #mouse movement param init
        self.lastCursorPos = QtCore.QPointF(10.0, 10.0)
        self.moving = False

        #view scene size
        self.viewportSize = QtCore.QSize(1600, 900)
        self.viewportMid = QtCore.QPointF(800, 450)
        self.viewRect = QtCore.QRect(0, 0, 1600, 900)

        #init openslide image level
        self.read_level = -1
        self.initial_readlevel = 0

        #cursor signal
        self.cursorSignal = CursorSignals()
    
    def reset(self):
        try:
            self.slide.close() 
        except Exception:
            pass
        self.slide = None
        self.moving = False
    
    def setGraphicsView(self, display):
        self.display = display
    
    def setNewImage(self, iImg):
        self.reset()

        self.slide = iImg
        self.viewRect = QtCore.QRect(0, 0, self.viewportSize.width(), self.viewportSize.height())
        #diagonal distance
        distance = 1e100
        
        #check the diagonal distance for different slide level
        #initial the highest resolution slide image first
        for r in range(self.slide.level_count):
            rDistance = (self.slide.level_dimensions[r][1] - self.viewRect.width())**2 + (self.slide.level_dimensions[r][0] - self.viewRect.height())**2
            if rDistance < distance:
                distance = rDistance
                self.read_level = r
                self.initial_readlevel = r
        
        for item in self.items():
            if item != self.slidePixmap:
                self.removeItem(item)
        self.updateScene()

    #read a specific region of the slide image
    def updateScene(self):
        if not self.slide:
            return
        tempImg = self.slide.read_region((self.viewRect.x(), self.viewRect.y()), self.read_level, (self.viewRect.width(), self.viewRect.height()))
        #convert colorspace, RGBA to ARGB
        #the ouput colorspace for slide image is RGBA and QImage colorspace format is ARGB
        qIm = ImageQt(tempImg)
        self.slidePixmap.setPixmap(QPixmap.fromImage(qIm))

    #update the ViewRect and reshow the new image region
    def updateViewRect(self, newTopLeft):
        if not self.slide:
            return
        dim = self.slide.level_dimensions[self.read_level]
        ds = self.slide.level_downsamples[self.read_level]

        #ensure the Topleft corner is in the valid boundary 
        newX = newTopLeft.x()
        newX = min(newX, (dim[0] - self.viewRect.width()) * ds)
        newX = max(newX, 0)

        newY = newTopLeft.y()
        newY = min(newY, (dim[1] - self.viewRect.height()) * ds)
        newY = max(newY, 0)

        self.viewRect = QtCore.QRect(newX, newY, self.viewRect.width(), self.viewRect.height())  
        self.updateScene()

    #change resolution function
    def zoom_in(self):
        if not self.slide:
            return
        decrement = self.read_level - 1
        if decrement >= 0:
            self.read_level = decrement
            #openslide level downsamples: a list of downsample factors for each level of the slide
            dPos = self.viewRect.topLeft() + self.viewportMid * self.slide.level_downsamples[
                self.read_level + 1] \
                    - self.viewportMid * self.slide.level_downsamples[self.read_level]
            self.updateViewRect(dPos)  

    def zoom_out(self):
        if not self.slide:
            return
        increment = self.read_level + 1
        if increment < self.slide.level_count:
            self.read_level = increment
            dPos = self.viewRect.topLeft() + self.viewportMid * self.slide.level_downsamples[
                self.read_level - 1] \
                    - self.viewportMid * self.slide.level_downsamples[self.read_level]
            self.updateViewRect(dPos)
    
    def moveleft(self):
        dPos = self.viewRect.topLeft() + QtCore.QPointF(20.0, 20.0) * self.slide.level_downsamples[self.read_level]
        self.updateViewRect(dPos)

    def moveright(self):
        dPos = self.viewRect.topLeft() - QtCore.QPointF(20.0, 20.0) * self.slide.level_downsamples[self.read_level]
        self.updateViewRect(dPos)

    def mousePressEvent(self, event):
        self.cursorSignal.press.emit()
        self.moving = True

    def mouseReleaseEvent(self, event):
        self.cursorSignal.release.emit()
        self.moving = False

    def mouseMoveEvent(self, event):
        assert isinstance(event, QGraphicsSceneMouseEvent)
        if self.slide and self.moving:
            dPos = self.viewRect.topLeft() - self.slide.level_downsamples[self.read_level] * (event.screenPos() - event.lastScreenPos())
            self.updateViewRect(dPos)

    def wheelEvent(self, event):
        if not self.slide:
            return
        if sys.platform is not 'darwin':
            if event.delta() < 0:
                self.zoom_in()
            else:
                self.zoom_out()