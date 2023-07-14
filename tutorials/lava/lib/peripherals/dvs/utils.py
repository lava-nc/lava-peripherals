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
