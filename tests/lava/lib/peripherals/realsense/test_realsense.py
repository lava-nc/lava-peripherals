import unittest
import numpy as np
import sys

try:
    import pyrealsense2 as rs

    has_valid_camera = False
    for device in rs.context().devices:
        color_sensor = device.first_color_sensor()
        depth_sensor = device.first_depth_sensor()

        if color_sensor is not None and depth_sensor is not None:
            has_valid_camera = True
            break
except ImportError:
    print("Need `pyrealsense2` library installed.", file=sys.stderr)
    exit(1)

from lava.magma.core.process.process import LogConfig

from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous


class TestRealsense(unittest.TestCase):
    @unittest.skipUnless(has_valid_camera, "Requires valid Realsense camera.")
    def test_init_camera(self) -> None:
        """"""
        pass

    def test_init_files(self) -> None:
        """"""
        pass

    def test_init_files_invalid_directory_path(self) -> None:
        """"""
        pass

    def test_init_files_invalid_png_prefix(self) -> None:
        """"""
        pass

    def test_init_files_invalid_exr_prefix(self) -> None:
        """"""
        pass


class TestLoihiDensePyRealsensePM(unittest.TestCase):
    @unittest.skipUnless(has_valid_camera, "Requires valid Realsense camera.")
    def test_run_camera(self) -> None:
        """"""
        pass

    def test_run_files(self) -> None:
        """"""
        pass


if __name__ == "__main__":
    unittest.main()
