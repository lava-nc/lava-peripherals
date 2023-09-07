# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
import os
import cv2

import pyrealsense2 as rs

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class DirectRealsenseInput(AbstractProcess):
    """Process to align BGR and depth frames obtained from the RealSense camera
    using RealSense SDK and then outputting them.

    Parameters
    ----------
    height: int
        Height of the image frames.
    width: int
        Width of the image frames.
    filename: str
        Path to directory that contains the images.
        If filename is not provided, it means using the camera
    """

    def __init__(
        self,
        height: int,
        width: int,
        filename: str = "",
    ) -> None:
        super().__init__(
            height=height,
            width=width,
            filename=filename,
        )
        self.color_frame_out = OutPort(shape=(height, width, 3))
        self.depth_frame_out = OutPort(shape=(height, width))


@implements(proc=DirectRealsenseInput, protocol=LoihiProtocol)
@requires(CPU)
class DirectRealsenseInputPM(PyLoihiProcessModel):
    color_frame_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    depth_frame_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._height = proc_params["height"]
        self._width = proc_params["width"]
        self._filename = proc_params["filename"]
        self._cur_steps = 0
        if self._filename == "":
            self.pipeline = rs.pipeline()
            config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                # Depth alignment needs a color frame. If a depth camera doesn't
                # have a color sensor, the alignment can't work properly.
                print("Don't get color sensor,\
                       the alignment can't work properly.")
                exit(0)
            config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

    def get_image_data(self):
        # using the camera
        if self._filename == "":
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            while not aligned_depth_frame or not color_frame:
                print("aligned_depth_frame or not color_frame not ready, \
                      waiting...")
            depth_image = np.array(aligned_depth_frame.get_data())
            color_image = np.array(color_frame.get_data())
        # using the recording file
        else:
            color_img_path = self._filename + f"color_{self._cur_steps}.png"
            depth_img_path = self._filename + f"depth_{self._cur_steps}.exr"
            color_image = cv2.imread(color_img_path)
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            depth_image = cv2.imread(depth_img_path,
                                     cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return color_image, depth_image

    def run_spk(self):
        self._cur_steps += 1
        color_image, depth_image = self.get_image_data()
        self.color_frame_out.send(color_image)
        self.depth_frame_out.send(depth_image)

    def _stop(self):
        self.pipeline.stop()
        super()._stop()
