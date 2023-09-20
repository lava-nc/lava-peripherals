import unittest
import numpy as np
import sys

from lava.lib.peripherals.realsense.realsense import Realsense

from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous

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
        """Test that the Realsense Process (reading from camera, aligning
        Depth frames to BGR frames) is instantiated correctly."""
        realsense = Realsense()

        self.assertEqual(realsense.bgr_out_port.shape[:2],
                         realsense.depth_out_port.shape)

    @unittest.skipUnless(has_valid_camera, "Requires valid Realsense camera.")
    def test_init_camera_not_align_depth_to_bgr(self) -> None:
        """Test that the Realsense Process (reading from camera, not aligning
        Depth frames to BGR frames) is instantiated correctly."""
        realsense = Realsense(align_depth_to_bgr=False)

        if realsense.bgr_2d_shape != realsense.depth_2d_shape:
            self.assertNotEqual(realsense.bgr_out_port.shape[:2],
                                realsense.depth_out_port.shape)

    @unittest.skip("align_depth_to_rgb not implemented for files.")
    def test_init_files(self) -> None:
        """Test that the Realsense Process (reading from files, aligning
        Depth frames to BGR frames) is instantiated correctly."""
        realsense = Realsense(directory_path="recording",
                              png_prefix="bgr_",
                              exr_prefix="depth_")

        self.assertEqual(realsense.bgr_out_port.shape, (480, 640, 3))
        self.assertEqual(realsense.depth_out_port.shape, (480, 640))

    def test_init_files_not_align_depth_to_bgr(self) -> None:
        """Test that the Realsense Process (reading from files, not aligning
        Depth frames to BGR frames) is instantiated correctly."""
        realsense = Realsense(align_depth_to_bgr=False,
                              directory_path="recording",
                              png_prefix="bgr_",
                              exr_prefix="depth_")

        self.assertEqual(realsense.bgr_out_port.shape, (480, 640, 3))
        self.assertEqual(realsense.depth_out_port.shape, (240, 320))

    def test_init_files_invalid_directory_path(self) -> None:
        """Test that initializing the Realsense Process with an invalid
        directory_path raises an NotADirectoryError"""
        with self.assertRaises(NotADirectoryError):
            _ = Realsense(directory_path="invalid")

    def test_init_files_invalid_png_prefix(self) -> None:
        """Test that initializing the Realsense Process with an invalid
        png_prefix raises a FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _ = Realsense(directory_path="recording",
                          png_prefix="invalid",
                          exr_prefix="depth_")

    def test_init_files_invalid_exr_prefix(self) -> None:
        """Test that initializing the Realsense Process with an invalid
        exr_prefix raises a FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _ = Realsense(directory_path="recording",
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
        realsense = Realsense(align_depth_to_bgr=False,
                              directory_path="recording",
                              png_prefix="bgr_",
                              exr_prefix="depth_")
        realsense.run(condition=RunSteps(num_steps=2), run_cfg=Loihi2SimCfg())
        realsense.stop()


if __name__ == "__main__":
    unittest.main()
