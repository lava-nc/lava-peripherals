# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import subprocess
import pyrealsense2 as rs
import numpy as np

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava.lib.peripherals.realsense.realsense import (
    DirectRealsenseInput,
    DirectRealsenseInputPM,
)


class TestPyRealSenseCameraModel(unittest.TestCase):
    def test_base_functionality_file(self):
        num_steps = 10
        height = 360
        width = 640
        rgbdcamera = DirectRealsenseInput(
            height,
            width)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        rgbdcamera.run(condition=run_condition, run_cfg=run_cfg)
        rgbdcamera.stop()


if __name__ == '__main__':
    unittest.main()
