# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from pathlib import Path
import sys
import os

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

try:
    import pyrealsense2 as rs
except ImportError:
    print("Need `pyrealsense2` library installed.", file=sys.stderr)
    exit(1)


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402


class RealSense(AbstractProcess):
    """Process that reads BGR+depth frames, either from a RealSense camera
    directly, or from recorded PNG+EXR files.

    If directory_path is not given, the Process will read and send BGR and
    depth frames from a RealSense camera.

    If directory_path is given, the Process will read and send BGR and depth
    frames from PNG and EXR files contained in the directory which the path
    points to.
    In this case, png_prefix and exr_prefix have to be provided to inform the
    Process on how the files are named.

    Parameters
    ----------
    align_depth_to_bgr: bool, optional
        Boolean flag controlling whether or not to align depth frames to BGR
        frames.
    directory_path: str, optional
        Path to directory that contains the images PNG+EXR files.
        If not given, will use the camera instead.
    png_prefix: str, optional
        Prefix of PNG files (recording of BGR frames).
    exr_prefix: str, optional
        Prefix of EXR files (recording of depth frames).
    """

    def __init__(self,
                 align_depth_to_bgr: ty.Optional[bool] = False,
                 directory_path: ty.Optional[str] = None,
                 png_prefix: ty.Optional[str] = "",
                 exr_prefix: ty.Optional[str] = "") -> None:
        super().__init__(
            align_depth_to_bgr=align_depth_to_bgr,
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
            bgr_sensor = device.first_color_sensor()
            depth_sensor = device.first_depth_sensor()

            if bgr_sensor is None:
                raise ValueError("No BGR sensor was found.")
            if depth_sensor is None:
                raise ValueError("No depth sensor was found.")

            bgr_stream = \
                profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_stream = \
                profile.get_stream(rs.stream.depth).as_video_stream_profile()

            self.bgr_2d_shape = (bgr_stream.height(), bgr_stream.width())
            self.bgr_3d_shape = self.bgr_2d_shape + (3,)
            self.depth_2d_shape = (depth_stream.height(), depth_stream.width())

            pipeline.stop()
        else:
            dir_path = Path(directory_path)

            if not dir_path.is_dir():
                raise NotADirectoryError(f"{dir_path} is not a directory.")

            bgr_sample_path = Path(directory_path) / f"{png_prefix}1.png"
            depth_sample_path = Path(directory_path) / f"{exr_prefix}1.exr"

            if not bgr_sample_path.is_file():
                raise FileNotFoundError(f"{bgr_sample_path} is not a file.")
            if not depth_sample_path.is_file():
                raise FileNotFoundError(f"{depth_sample_path} is not a file.")

            bgr_sample_frame = cv2.imread(str(bgr_sample_path))
            depth_sample_frame = \
                cv2.imread(str(depth_sample_path),
                           cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            self.bgr_3d_shape = bgr_sample_frame.shape
            self.bgr_2d_shape = self.bgr_3d_shape[:2]
            self.depth_2d_shape = depth_sample_frame.shape

            if align_depth_to_bgr:
                raise NotImplementedError("Aligning Depth frames to BGR "
                                          "frames is not implemented.")

        bgr_out_shape = self.bgr_3d_shape
        depth_out_shape = self.depth_2d_shape

        if align_depth_to_bgr:
            depth_out_shape = self.bgr_2d_shape

        self.proc_params["bgr_2d_shape"] = self.bgr_2d_shape
        self.proc_params["bgr_3d_shape"] = self.bgr_3d_shape
        self.proc_params["depth_2d_shape"] = self.depth_2d_shape

        self.bgr_out_port = OutPort(shape=bgr_out_shape)
        self.depth_out_port = OutPort(shape=depth_out_shape)


@implements(proc=RealSense, protocol=LoihiProtocol)
@requires(CPU)
class LoihiDensePyRealSensePM(PyLoihiProcessModel):
    bgr_out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    depth_out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)

        self._bgr_2d_shape = proc_params["bgr_2d_shape"]
        self._bgr_3d_shape = proc_params["bgr_3d_shape"]
        self._depth_2d_shape = proc_params["depth_2d_shape"]
        self._align_depth_to_bgr = proc_params["align_depth_to_bgr"]
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

    def run_spk(self) -> None:
        bgr_image, depth_image = self._get_frames()

        self.bgr_out_port.send(bgr_image)
        self.depth_out_port.send(depth_image)

    def _get_frames(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Get BGR and depth frames.

        If directory_path is None, get frames from camera.
        If directory_path is not None, get frames from files
        contained in the directory which the path points to.

        Returns
        ----------
        bgr_frame : np.ndarray
            BGR frame.
        depth_frame : np.ndarray
            Depth frame.
        """
        if self._directory_path is None:
            return self._get_frames_from_camera()
        else:
            return self._get_frames_from_directory()

    def _get_frames_from_camera(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Get BGR and depth frames from camera.

        Returns
        ----------
        bgr_frame : np.ndarray
            BGR frame.
        depth_frame : np.ndarray
            Depth frame.
        """
        frameset = self._pipeline.wait_for_frames()
        if self._align_depth_to_bgr:
            frameset = self._align.process(frameset)

        bgr_frame = np.array(frameset.get_color_frame().get_data())
        depth_frame = np.array(frameset.get_depth_frame().get_data())

        return bgr_frame, depth_frame

    def _get_frames_from_directory(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Get BGR and depth frames from files.

        Returns
        ----------
        bgr_frame : np.ndarray
            BGR frame.
        depth_frame : np.ndarray
            Depth frame.
        """
        bgr_frame_path = Path(self._directory_path) / f"{self._png_prefix}" \
                                                      f"{self.time_step}.png"
        depth_frame_path = Path(self._directory_path) / f"{self._exr_prefix}" \
                                                        f"{self.time_step}.exr"

        bgr_frame = cv2.imread(str(bgr_frame_path))
        depth_frame = cv2.imread(str(depth_frame_path),
                                 cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        return bgr_frame, depth_frame

    def _stop(self) -> None:
        if self._directory_path is None:
            self._pipeline.stop()

        super()._stop()
