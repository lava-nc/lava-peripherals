# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import cv2
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

import time


class OpenCVDisplay(AbstractProcess):
    """Process that receives datas from connected processmodel.
    The frame data which are needed to display from connected processmodels
    need a corresponding InPort which is connected to an OutPort.
    The number of input ports is decided by the number of
    process models connected to this opencv process model.
    """
    def __init__(self,
                 shape):
        super().__init__(shape=shape)
        self.frame_port = InPort(shape=shape)


@implements(proc=OpenCVDisplay, protocol=LoihiProtocol)
@requires(CPU)
class OpenCVDisplayPM(PyLoihiProcessModel):
    frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._shape = proc_params['shape']
        cv2.namedWindow("Image Display")

    def display(self, image):
        """Visualize images on OpenCV windows
        Takes a NumPy image array formatted as RGBA and sends to OpenCV for
        visualization.
        Parameters
        ----------
        image           [np.Array]: NumPy array of rs Image
        """
        image = np.array(np.sum(image, axis=2),
                         dtype=np.uint8)
        cv2.imshow("Image Display", image)
        cv2.waitKey(1)

    def run_spk(self):
        frame = self.frame_port.recv()
        self.display(frame)
