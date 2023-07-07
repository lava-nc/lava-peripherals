
import numpy as np
import time
import cv2

import os
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.io.sink import RingBuffer

from lava.lib.peripherals.dvs.process import PropheseeCamera
from metavision_core.utils import get_sample
from metavision_sdk_cv import TrailFilterAlgorithm

from multiprocessing import Pipe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import IPython


from metavision_core.event_io import RawReader
import tonic
from tonic.transforms import Compose
from metavision_ml.preprocessing import histo, histo_quantized
from metavision_core.event_io import RawReader

from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
from lava.proc.lif.process import LIF, LogConfig
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
import logging

from scipy.sparse import csr_matrix

class VisProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, shape):
        super().__init__(shape=shape)
        self.shape = shape
        self.s_in = InPort(shape=shape)



@implements(proc=VisProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyVisProcess(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params['shape']
        self.label = "live plot"
        self.max = 0

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        frame = self.s_in.recv()
        frame.sum(axis=0).sum(axis=0)
        data = np.zeros(self.shape[-2:] + (3, ), np.uint8) 

        data[:, :, 0] = 255 // (frame.max() + 1) * frame[0, 1, :, :].astype(np.uint8)
        data[:, :, 1] = 255 // (frame.max() + 1) * frame[0, 0, :, :].astype(np.uint8)
        data[:, :, 2] = 255 // (frame.max() + 1) * frame[0, 1, :, :].astype(np.uint8)
        
        cv2.imshow(self.label, data)
        cv2.waitKey(1)


    def _stop(self):
        cv2.destroyWindow(self.label)
        super()._stop()



class VisUpDownProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, shape):
        super().__init__(shape=shape)
        self.shape = shape
        self.frame_in = InPort(shape=shape)
        self.up_in = InPort(shape=shape)
        self.down_in = InPort(shape=shape)
        self.left_in = InPort(shape=shape)
        self.right_in = InPort(shape=shape)



@implements(proc=VisUpDownProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyVisUpDownProcess(PyLoihiProcessModel):
    frame_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    up_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    down_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    left_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    right_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params['shape']
        self.height = self.shape[2]
        self.width = self.shape[3]
        self.label = "live plot"

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        frame = self.frame_in.recv()
        up = self.up_in.recv().sum()
        down = self.down_in.recv().sum()
        left = self.left_in.recv().sum()
        right = self.right_in.recv().sum()
        # print(up, down, left, right)
        
        frame.sum(axis=0).sum(axis=0)
        img = np.zeros(self.shape[-2:] + (3, ), np.uint8) 

        # data[:, :, 0] = 255 // (frame.max() + 1) * frame[0, 1, :, :].astype(np.uint8)
        img[:, :, 1] = 255 // (frame.max() + 1) * frame[0, 0, :, :].astype(np.uint8)
        # data[:, :, 2] = 255 // (frame.max() + 1) * frame[0, 1, :, :].astype(np.uint8)

        ud_arrow_start = (self.width // 2, self.height // 2)
        ud_arrow_end = (self.width // 2, self.height // 2 + int(down - up))
        ud_arrow_end = np.clip(ud_arrow_end, (0, 0), (self.width, self.height))
        img = cv2.arrowedLine(img,
                              ud_arrow_start,
                              ud_arrow_end,
                              (0, 0, 255),
                              2)

        lr_arrow_start = (self.width // 2, self.height // 2)
        lr_arrow_end = (self.width // 2 + int(left-right), self.height // 2)
        lr_arrow_end = np.clip(lr_arrow_end, (0, 0), (self.width, self.height))

        img = cv2.arrowedLine(img,
                              lr_arrow_start,
                              lr_arrow_end,
                              (255, 0, 255),
                              2)
        
        cv2.imshow(self.label, img)
        cv2.waitKey(1)


    def _stop(self):
        cv2.destroyWindow(self.label)
        super()._stop()
