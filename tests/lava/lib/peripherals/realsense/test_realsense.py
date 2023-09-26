# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import sys
from pathlib import Path

from lava.lib.peripherals.realsense.realsense import Realsense
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

has_valid_camera = False
try:
    import pyrealsense2 as rs

    for device in rs.context().devices:
        color_sensor = device.first_color_sensor()
        depth_sensor = device.first_depth_sensor()

        if color_sensor is not None and depth_sensor is not None:
            has_valid_camera = True
            break
except ImportError:
    print("Need `pyrealsense2` library installed.", file=sys.stderr)
    exit(1)


class TestRealsense(unittest.TestCase):
    @unittest.skipUnless(has_valid_camera, "Requires valid Realsense camera.")
    def test_init_camera(self) -> None:
        """Test that the Realsense Process (reading from camera)
        is instantiated correctly."""
        realsense = Realsense()

        if realsense.bgr_2d_shape != realsense.depth_2d_shape:
            self.assertNotEqual(realsense.bgr_out_port.shape[:2],
                                realsense.depth_out_port.shape)

        realsense = Realsense(align_depth_to_bgr=True)

        self.assertEqual(realsense.bgr_out_port.shape[:2],
                         realsense.depth_out_port.shape)

    def test_init_files(self) -> None:
        """Test that the Realsense Process (reading from files)
        is instantiated correctly."""
        directory_path = Path(__file__).resolve().parent / "recording"

        # TODO: Uncomment this when align_depth_to_bgr=True is supported with
        #  files
        # realsense = Realsense(align_depth_to_bgr=True,
        #                       directory_path=str(directory_path),
        #                       png_prefix="bgr_",
        #                       exr_prefix="depth_")
        #
        # self.assertEqual(realsense.bgr_out_port.shape, (480, 640, 3))
        # self.assertEqual(realsense.depth_out_port.shape, (480, 640))

        realsense = Realsense(directory_path=str(directory_path),
                              png_prefix="bgr_",
                              exr_prefix="depth_")

        self.assertEqual(realsense.bgr_out_port.shape, (480, 640, 3))
        self.assertEqual(realsense.depth_out_port.shape, (240, 320))

    def test_init_files_invalid_parameters(self) -> None:
        """Test that initializing the Realsense Process (reading from files)
        with invalid parameters raise errors:"""
        with self.assertRaises(NotADirectoryError):
            _ = Realsense(directory_path="invalid")

        directory_path = Path(__file__).resolve().parent / "recording"

        with self.assertRaises(FileNotFoundError):
            _ = Realsense(directory_path=str(directory_path),
                          png_prefix="invalid",
                          exr_prefix="depth_")

        with self.assertRaises(FileNotFoundError):
            _ = Realsense(directory_path=str(directory_path),
                          png_prefix="bgr_",
                          exr_prefix="invalid")


class TestLoihiDensePyRealsensePM(unittest.TestCase):
    @unittest.skipUnless(has_valid_camera, "Requires valid Realsense camera.")
    def test_run_camera(self) -> None:
        """Test that the Realsense Process (reading from camera) runs
        correctly."""
        realsense = Realsense()
        realsense.run(condition=RunSteps(num_steps=2), run_cfg=Loihi2SimCfg())
        realsense.stop()

    def test_run_files(self) -> None:
        """Test that the Realsense Process (reading from files) runs
        correctly."""
        directory_path = Path(__file__).resolve() / "recording"

        realsense = Realsense(directory_path=str(directory_path),
                              png_prefix="bgr_",
                              exr_prefix="depth_")
        realsense.run(condition=RunSteps(num_steps=2), run_cfg=Loihi2SimCfg())
        realsense.stop()


if __name__ == "__main__":
    unittest.main()
