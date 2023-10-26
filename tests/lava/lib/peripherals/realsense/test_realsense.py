# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import sys
from pathlib import Path

from lava.lib.peripherals.realsense.realsense import RealSense
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

USE_CAMERA_TESTS = False
try:
    import pyrealsense2 as rs

    for device in rs.context().devices:
        color_sensor = device.first_color_sensor()
        depth_sensor = device.first_depth_sensor()

        if color_sensor is not None and depth_sensor is not None:
            USE_CAMERA_TESTS = True
            break
except ImportError:
    print("Need `pyrealsense2` library installed.", file=sys.stderr)
    exit(1)


class TestRealSense(unittest.TestCase):
    @unittest.skipUnless(USE_CAMERA_TESTS,
                         "Requires a Realsense camera to be connected.")
    def test_init_camera(self) -> None:
        """Test that the Realsense Process is instantiated correctly when
        reading from camera."""
        realsense = RealSense()

        if realsense.bgr_2d_shape != realsense.depth_2d_shape:
            self.assertNotEqual(realsense.bgr_out_port.shape[:2],
                                realsense.depth_out_port.shape)

        realsense = RealSense(align_depth_to_bgr=True)

        self.assertEqual(realsense.bgr_out_port.shape[:2],
                         realsense.depth_out_port.shape)

    def test_init_files(self) -> None:
        """Test that the Realsense Process is instantiated correctly when
        reading from files."""
        directory_path = Path(__file__).resolve().parent / "recording"

        realsense = RealSense(directory_path=str(directory_path),
                              png_prefix="bgr_",
                              exr_prefix="depth_")

        self.assertEqual(realsense.bgr_out_port.shape, (480, 640, 3))
        self.assertEqual(realsense.depth_out_port.shape, (240, 320))

    def test_init_files_invalid_parameters(self) -> None:
        """Test that initializing the Realsense Process reading from
        files, with invalid parameters, raises errors."""
        with self.assertRaises(NotADirectoryError):
            _ = RealSense(directory_path="invalid")

        directory_path = Path(__file__).resolve().parent / "recording"

        with self.assertRaises(FileNotFoundError):
            _ = RealSense(directory_path=str(directory_path),
                          png_prefix="invalid",
                          exr_prefix="depth_")

        with self.assertRaises(FileNotFoundError):
            _ = RealSense(directory_path=str(directory_path),
                          png_prefix="bgr_",
                          exr_prefix="invalid")


class TestLoihiDensePyRealSensePM(unittest.TestCase):
    @unittest.skipUnless(USE_CAMERA_TESTS,
                         "Requires a Realsense camera to be connected.")
    def test_run_camera(self) -> None:
        """Test that the Realsense Process runs correctly when reading from
        camera."""
        realsense = RealSense()
        realsense.run(condition=RunSteps(num_steps=2), run_cfg=Loihi2SimCfg())
        realsense.stop()

    def test_run_files(self) -> None:
        """Test that the Realsense Process runs correctly when reading from
        files."""
        directory_path = Path(__file__).resolve().parent / "recording"

        realsense = RealSense(directory_path=str(directory_path),
                              png_prefix="bgr_",
                              exr_prefix="depth_")
        realsense.run(condition=RunSteps(num_steps=2), run_cfg=Loihi2SimCfg())
        realsense.stop()


if __name__ == "__main__":
    unittest.main()
