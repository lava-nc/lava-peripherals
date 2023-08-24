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
    """Process that receives (1) the raw rs events, (2) the spike rates
    of the selective as well as (3) the multi-peak DNF per pixel. It sends
    these values through a multiprocessing pipe (rather than a Lava OutPort)
    to allow for plotting."
    """
    def __init__(self,
                 shape_rs_frame,
                 shape_dnf,
                 plot_base_width,
                 data_shape):
        super().__init__(shape_rs_frame=shape_rs_frame,
                         shape_dnf=shape_dnf,
                         plot_base_width=plot_base_width,
                         data_shape=data_shape)
        self.rs_frame_port = InPort(shape=shape_rs_frame)
        self.dnf_multipeak_rates_port = InPort(shape=shape_dnf)
        self.dnf_selective_rates_port = InPort(shape=shape_dnf)


@implements(proc=OpenCVDisplay, protocol=LoihiProtocol)
@requires(CPU)
class OpenCVDisplayPM(PyLoihiProcessModel):
    rs_frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    dnf_multipeak_rates_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    dnf_selective_rates_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._plot_base_width = proc_params['plot_base_width']
        self._data_shape = proc_params['data_shape']
        # Create windows
        cv2.namedWindow("Realsense File Input")
        cv2.namedWindow("DNF Multi-Peak (spike rates)")
        cv2.namedWindow("DNF Selective (spike rates)")
        self.prev_time = 0

    def display(self, rs_image, dnf_multipeak_image, dnf_selective_image):
        """Visualize images on OpenCV windows

        Takes a NumPy image array formatted as RGBA and sends to OpenCV for
        visualization.

        Parameters
        ----------
        rs_image           [np.Array]: NumPy array of rs Image
        dnf_multipeak_image [np.Array]: NumPy array of DNF Multi-Peak
        dnf_selective_image [np.Array]: NumPy array of DNF Selective
        """
        # rs_image_bgr = (rs_image * 255).astype(np.uint8)
        # dnf_multipeak_bgr = (dnf_multipeak_image * 255).astype(np.uint8)
        # dnf_selective_bgr = (dnf_selective_image * 255).astype(np.uint8)

        img_shape = (rs_image.shape[0], rs_image.shape[1],)
        num_bins = rs_image.shape[2]
        print("img_shape", img_shape)
        print("roscam_frame_ds_image", rs_image.shape)
        print("dnf_multipeak_rates_ds_image", dnf_multipeak_image.shape)
        print("dnf_selective_rates_ds_image", dnf_selective_image.shape)

        roscam_frame_ds_image_new = np.array(np.sum(rs_image, axis=2),
                                             dtype=np.uint8)
        dnf_multipeak_rates_ds_image_new = np.array(np.sum(dnf_multipeak_image,
                                                           axis=2),
                                                    dtype=np.uint8)
        dnf_selective_rates_ds_image_new = np.array(np.sum(dnf_selective_image,
                                                           axis=2),
                                                    dtype=np.uint8)
        cv2.imshow("Realsense File Input", roscam_frame_ds_image_new)
        cv2.imshow("DNF Multi-Peak (spike rates)",
                   dnf_multipeak_rates_ds_image_new)
        cv2.imshow("DNF Selective (spike rates)",
                   dnf_selective_rates_ds_image_new)
        cv2.waitKey(1)
        print(f"Time to display (seconds): \
             {(time.time_ns()-self.prev_time) / 1e9}")
        self.prev_time = time.time_ns()

    def run_spk(self):
        rs_frame = self.rs_frame_port.recv()
        dnf_multipeak_rates = self.dnf_multipeak_rates_port.recv()
        dnf_selective_rates = self.dnf_selective_rates_port.recv()

        # rs_frame_ds_image = np.rot90(rs_frame, -1)
        # dnf_multipeak_rates_ds_image = np.rot90(dnf_multipeak_rates, -1)
        # dnf_selective_rates_ds_image = np.rot90(dnf_selective_rates, -1)

        self.display(rs_frame, dnf_multipeak_rates, dnf_selective_rates)
