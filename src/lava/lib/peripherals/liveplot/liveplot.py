# Copyright (C) 2023 Intel Corporation
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


class OpenCVDisplay(AbstractProcess):
    """ Process that display incoming image with OpenCV.

    This process continuously display grayscale/RGB images coming through its
    InPort. The number of input ports is decided by the number of process models
    connected to this opencv process model.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort.
        (height, width) or (height, width, 1) for grayscale images.
        (height, width, 3) for RGB images.
    opencv_window_name: str
        Name of OpenCV window.
    """
    def __init__(self,
                 shape: tuple,
                 opencv_window_name: str = "Image Display"):
        super().__init__(shape=shape,
                         opencv_window_name=opencv_window_name)
        if len(shape) not in (2, 3):
            raise ValueError("Shape must be in the format (height, width) for \
                             grayscale or (height, width, 3) for RGB.")
        if len(shape) == 3 and shape[2] not in (1, 3):
            raise ValueError("For a 3-channel image, shape must be \
                             (height, width, 3). For grayscale, use \
                             (height, width) or (height, width, 1).")
        self.frame_port = InPort(shape=shape)


@implements(proc=OpenCVDisplay, protocol=LoihiProtocol)
@requires(CPU)
class OpenCVDisplayPM(PyLoihiProcessModel):
    frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._shape = proc_params['shape']
        self._window_name = proc_params['opencv_window_name']
        cv2.namedWindow(self._window_name)

    def display(self, image):
        """ Visualize images on OpenCV windows.

        Takes a NumPy image array formatted as RGBA and sends to OpenCV for
        visualization. RGBA image will be converted to grayscale by summing
        the color channels. If it's a 2D image, no changes will be made.

        Parameters
        ----------
        image: np.ndarray
            Image to display.
        """
        image = np.array(np.sum(image, axis=2),
                         dtype=np.uint8)
        cv2.imshow(self._window_name, image)
        cv2.waitKey(1)

    def run_spk(self):
        frame = self.frame_port.recv()
        self.display(frame)

    def _stop(self):
        cv2.destroyAllWindow()
        super()._stop()
