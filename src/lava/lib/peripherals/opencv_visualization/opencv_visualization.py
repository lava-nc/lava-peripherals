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
    """Process that receives datas from connected processmodel. It sends
    these values through a multiprocessing pipe (rather than a Lava OutPort)
    to allow for plotting. The number of input ports decided by the number of
    process models connected to this opencv process model"
    """
    def __init__(self,
                 shape_frame,
                 plot_base_width,
                 data_shape):
        super().__init__(shape_frame=shape_frame,
                         plot_base_width=plot_base_width,
                         data_shape=data_shape)
        # initialize some input ports as needed.
        self.frame_port = InPort(shape=shape_frame)


@implements(proc=OpenCVDisplay, protocol=LoihiProtocol)
@requires(CPU)
class OpenCVDisplayPM(PyLoihiProcessModel):
    frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._plot_base_width = proc_params['plot_base_width']
        self._data_shape = proc_params['data_shape']
        # Create windows
        cv2.namedWindow("Input image")
        self.prev_time = 0

    def display(self, image):
        """Visualize images on OpenCV windows
        Takes a NumPy image array formatted as RGBA and sends to OpenCV for
        visualization.
        Parameters
        ----------
        image           [np.Array]: NumPy array of rs Image
        """

        img_shape = (image.shape[0], image.shape[1],)
        print("img_shape", img_shape)
        print("roscam_frame_ds_image", image.shape)

        cam_frame_image_new = np.array(np.sum(image, axis=2),
                                       dtype=np.uint8)
        cv2.imshow("Input image", cam_frame_image_new)
        cv2.waitKey(1)
        print(f"Time to display (seconds): \
             {(time.time_ns()-self.prev_time) / 1e9}")
        self.prev_time = time.time_ns()

    def run_spk(self):
        frame = self.frame_port.recv()
        self.display(frame)
