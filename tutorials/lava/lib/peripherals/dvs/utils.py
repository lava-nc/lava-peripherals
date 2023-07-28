import numpy as np
import time
import cv2
import scipy

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

from metavision_core.utils import get_sample
from metavision_sdk_cv import TrailFilterAlgorithm

from multiprocessing import Pipe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import IPython


from metavision_core.event_io import RawReader
from metavision_ml.preprocessing import histo, histo_quantized
from metavision_core.event_io import RawReader

from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
from lava.proc.lif.process import LIF, LogConfig
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
import logging

from scipy.sparse import csr_matrix


class EventVisualizer(AbstractProcess):
    """
    Process that receives arbitrary vectors and visualizes them using CV.

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, shape):
        super().__init__(shape=shape)
        self.shape = shape
        self.s_in = InPort(shape=shape)


@implements(proc=EventVisualizer, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyEventVisualizerModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params["shape"]
        self.label = "live plot"
        self.max = 0

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        frame = self.s_in.recv()
        frame.sum(axis=0).sum(axis=0)
        data = np.zeros(self.shape[-2:] + (3,), np.uint8)

        data[:, :, 0] = (
            255 // (frame.max() + 1) * frame[0, 1, :, :].astype(np.uint8)
        )
        data[:, :, 1] = (
            255 // (frame.max() + 1) * frame[0, 0, :, :].astype(np.uint8)
        )
        data[:, :, 2] = (
            255 // (frame.max() + 1) * frame[0, 1, :, :].astype(np.uint8)
        )

        cv2.imshow(self.label, data)
        cv2.waitKey(1)

    def _stop(self):
        cv2.destroyWindow(self.label)
        super()._stop()


class VisSwipeProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, shape, direction_shape):
        super().__init__(shape=shape,
                         direction_shape=direction_shape)
        self.shape = shape
        self.direction_shape = direction_shape
        self.frame_in = InPort(shape=shape)
        self.left_in = InPort(shape=shape)
        self.right_in = InPort(shape=shape)
        self.direction_in = InPort(shape=direction_shape)

        # self.up_in = InPort(shape=direction_shape)
        # self.down_in = InPort(shape=direction_shape)
        # self.left_in = InPort(shape=direction_shape)
        # self.right_in = InPort(shape=direction_shape)


@implements(proc=VisSwipeProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyVisUpDownProcess(PyLoihiProcessModel):
    frame_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    direction_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    
    # up_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    # down_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    left_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    right_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params["shape"]
        self.direction_shape = proc_params["direction_shape"]
        self.height = self.shape[2]
        self.width = self.shape[3]
        self.label = "live plot"
        self.left_img = "left"
        self.right_img = "right"

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        frame = self.frame_in.recv()
        frame_left = self.left_in.recv()
        frame_right = self.right_in.recv()
        direction_in = self.direction_in.recv()
        n_per_direction = self.direction_shape[0] // 2
        # up = direction_in[:n_per_direction].sum()
        # down = direction_in[n_per_direction:2*n_per_direction].sum()
        # left = direction_in[2*n_per_direction:3*n_per_direction].sum()
        # right = direction_in[3*n_per_direction:].sum()

        left = direction_in[:n_per_direction].sum()
        right = direction_in[n_per_direction:2*n_per_direction].sum()
        # print(left, right)

        frame = frame.sum(axis=0).sum(axis=0)
        img = np.zeros(frame.shape + (3,), np.uint8)
        img[:, :, 1] = 255 // (frame.max() + 1) * frame.astype(np.uint8)

        frame_left = frame_left.sum(axis=0).sum(axis=0)
        img_left = np.zeros(frame_left.shape + (3,), np.uint8)
        img_left[:, :, 1] = 255 // (frame_left.max() + 1) * frame_left.astype(np.uint8)

        frame_right = frame_right.sum(axis=0).sum(axis=0)
        img_right = np.zeros(frame_right.shape + (3,), np.uint8)
        img_right[:, :, 1] = 255 // (frame_right.max() + 1) * frame_right.astype(np.uint8)

        # ud_arrow_start = (self.width // 2, self.height // 2)
        # ud_arrow_end = (
        #     self.width // 2,
        #     self.height // 2 + int(up-down) * 2,
        # )
        # ud_arrow_end = np.clip(ud_arrow_end, (0, 0), (self.width, self.height))

        # img = cv2.arrowedLine(
        #     img, ud_arrow_start, ud_arrow_end, (0, 0, 255), 2
        # )

        lr_arrow_start = (self.width // 2, self.height // 2)
        lr_arrow_end = (
            self.width // 2 + int(right - left) * 3,
            self.height // 2,
        )
        lr_arrow_end = np.clip(lr_arrow_end, (0, 0), (self.width, self.height))

        img = cv2.arrowedLine(
            img, lr_arrow_start, lr_arrow_end, (255, 0, 255), 2
        )

        cv2.imshow(self.label, img)
        cv2.moveWindow(self.label, 2400, 0)
        cv2.waitKey(1)
        
        cv2.imshow(self.left_img, img_left)
        cv2.moveWindow(self.left_img, 2000, 0)
        cv2.waitKey(1)
        
        cv2.imshow(self.right_img, img_right)
        cv2.moveWindow(self.right_img, 2800, 0)
        cv2.waitKey(1)

    def _stop(self):
        cv2.destroyAllWindows()
        super()._stop()
