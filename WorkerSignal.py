from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time

'''
Thread IO
for passing the data between different worker thread
'''
class CursorSignals(QObject):
    press = pyqtSignal()
    release = pyqtSignal()

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    
    Supported signals are:

    finished
        No data

    error
        'tuple'

    result
        'object'
    '''
    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    '''
    Worker thread

    :param args: Arguments to make avaiable to the run code
    :param kwargs: Keywords arguments to make available to the run code

    possible to pass the function objection to the Worker class

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
    @pyqtSlot()
    def run(self):
        '''
        Your code goes in this function
        '''
        self.signals.started.emit()

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
