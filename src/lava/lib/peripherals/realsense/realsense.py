# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import os
import cv2
import typing as ty
from pathlib import Path

import pyrealsense2 as rs

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class Realsense(AbstractProcess):
    """Process that reads RGB+Depth frames, either from a Realsense camera
    directly, or from recorded PNG+EXR files.

    If directory_path is not given, the Process will read and send RGB and
    Depth frames from a Realsense camera.

    If directory_path is given, the Process will read and send RGB and Depth
    frames from PNG and EXR files contained in the directory which the path
    points to.
    In this case, png_prefix and exr_prefix have to be provided to inform the
    Process on how the files are named.

    Parameters
    ----------
    align_depth_to_rgb: bool, optional
        Boolean flag controlling whether or not to align Depth frames to RGB
        frames.
    directory_path: str, optional
        Path to directory that contains the images PNG+EXR files.
        If not given, will use the camera instead.
    png_prefix: str, optional
        Prefix of PNG files (recording of RGB frames).
    exr_prefix: str, optional
        Prefix of EXR files (recording of Depth frames).
    """
    def __init__(self,
                 align_depth_to_rgb: bool = True,
                 directory_path: ty.Optional[str] = None,
                 png_prefix: ty.Optional[str] = None,
                 exr_prefix: ty.Optional[str] = None) -> None:
        super().__init__(
            align_depth_to_rgb=align_depth_to_rgb,
            directory_path=directory_path,
            png_prefix=png_prefix,
            exr_prefix=exr_prefix
        )

        if directory_path is None:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth)
            config.enable_stream(rs.stream.color)

            profile = pipeline.start(config)
            device = profile.get_device()
            rgb_sensor = device.first_color_sensor()
            depth_sensor = device.first_depth_sensor()

            if rgb_sensor is None:
                raise ValueError("No RGB sensor was found.")
            if depth_sensor is None:
                raise ValueError("No Depth sensor was found.")

            rgb_stream = \
                profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_stream = \
                profile.get_stream(rs.stream.depth).as_video_stream_profile()

            self.rgb_2d_shape = (rgb_stream.height(), rgb_stream.width())
            self.rgb_3d_shape = self.rgb_2d_shape + (3,)
            self.depth_2d_shape = (depth_stream.height(), depth_stream.width())

            self.proc_params["rgb_2d_shape"] = self.rgb_2d_shape
            self.proc_params["rgb_3d_shape"] = self.rgb_3d_shape
            self.proc_params["depth_2d_shape"] = self.depth_2d_shape

            rgb_out_shape = self.rgb_3d_shape
            depth_out_shape = self.depth_2d_shape

            if align_depth_to_rgb:
                depth_out_shape = self.rgb_2d_shape

            pipeline.stop()
        else:
            dir_path = Path(directory_path)

            if not dir_path.is_dir():
                raise ValueError(f"Directory {dir_path} does not exist.")

            if png_prefix is None or exr_prefix is None:
                raise ValueError(f"<png_prefix> and <exr_prefix> have to be "
                                 f"provided when <directory_path> is provided.")

            rgb_sample_path = Path(directory_path) / f"{png_prefix}1.png"
            depth_sample_path = Path(directory_path) / f"{exr_prefix}1.exr"

            rgb_sample_frame = cv2.imread(rgb_sample_path)
            depth_sample_frame = \
                cv2.imread(depth_sample_path,
                           cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            self.rgb_3d_shape = rgb_sample_frame.shape
            self.rgb_2d_shape = self.rgb_3d_shape[:2]
            self.depth_2d_shape = depth_sample_frame

            self.proc_params["rgb_2d_shape"] = self.rgb_2d_shape
            self.proc_params["rgb_3d_shape"] = self.rgb_3d_shape
            self.proc_params["depth_2d_shape"] = self.depth_2d_shape

            rgb_out_shape = self.rgb_3d_shape
            depth_out_shape = self.depth_2d_shape

        self.rgb_out_port = OutPort(shape=rgb_out_shape)
        self.depth_out_port = OutPort(shape=depth_out_shape)


@implements(proc=Realsense, protocol=LoihiProtocol)
@requires(CPU)
class LoihiDensePyRealsensePM(PyLoihiProcessModel):
    rgb_out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    depth_out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

        self._rgb_2d_shape = proc_params["rgb_2d_shape"]
        self._rgb_3d_shape = proc_params["rgb_3d_shape"]
        self._depth_2d_shape = proc_params["depth_2d_shape"]
        self._align_depth_to_rgb = proc_params["align_depth_to_rgb"]
        self._directory_path = proc_params["directory_path"]
        self._png_prefix = proc_params["png_prefix"]
        self._exr_prefix = proc_params["exr_prefix"]

        if self._directory_path is None:
            self._pipeline = rs.pipeline()

            config = rs.config()
            config.enable_stream(rs.stream.depth)
            config.enable_stream(rs.stream.color)

            pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device.hardware_reset()

            self._pipeline.start(config)
            self._align = rs.align(rs.stream.color)
        else:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    def run_spk(self) -> None:
        rgb_image, depth_image = self._get_frames()

        self.rgb_out_port.send(rgb_image)
        self.depth_out_port.send(depth_image)

    def _get_frames(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Get RGB and Depth frames.

        If directory_path is None, get frames from camera.
        If directory_path is not None, get frames from files
        contained in the directory which the path points to.

        Returns
        ----------
        rgb_frame : np.ndarray
            RGB frame.
        depth_frame : np.ndarray
            Depth frame.
        """
        if self._directory_path is None:
            return self._get_frames_from_camera()
        else:
            return self._get_frames_from_directory()

    def _get_frames_from_camera(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Get RGB and Depth frames from camera.

        Returns
        ----------
        rgb_frame : np.ndarray
            RGB frame.
        depth_frame : np.ndarray
            Depth frame.
        """
        frameset = self._pipeline.wait_for_frames()
        if self._align_depth_to_rgb:
            frameset = self._align.process(frameset)

        rgb_frame = np.array(frameset.get_color_frame().get_data())
        depth_frame = np.array(frameset.get_depth_frame().get_data())

        return rgb_frame, depth_frame

    def _get_frames_from_directory(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Get RGB and Depth frames from files.

        Returns
        ----------
        rgb_frame : np.ndarray
            RGB frame.
        depth_frame : np.ndarray
            Depth frame.
        """
        rgb_frame_path = Path(self._directory_path) / f"{self._png_prefix}" \
                                                      f"{self.time_step}.png"
        depth_frame_path = Path(self._directory_path) / f"{self._exr_prefix}" \
                                                        f"{self.time_step}.exr"

        rgb_frame = cv2.imread(str(rgb_frame_path))
        depth_frame = cv2.imread(str(depth_frame_path),
                                 cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        return rgb_frame, depth_frame

    def _stop(self) -> None:
        if self._directory_path is None:
            self._pipeline.stop()

        super()._stop()
