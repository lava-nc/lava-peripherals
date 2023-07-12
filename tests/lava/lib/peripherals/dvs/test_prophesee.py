# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

import numpy as np
import os
import time
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous

from lava.lib.peripherals.dvs.prophesee import PropheseeCamera, PyPropheseeCameraModel
from lava.lib.peripherals.dvs.transform import Compose, Downsample
from metavision_core.utils import get_sample
from metavision_core.event_io import RawReader
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm

SEQUENCE_FILENAME_RAW = "sparklers.raw"
get_sample(SEQUENCE_FILENAME_RAW)
assert os.path.isfile(SEQUENCE_FILENAME_RAW)


class Recv(AbstractProcess):
    """Process that receives arbitrary dense data and stores it in a Var.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Var.
    buffer_size: optional, int
        Size of buffer storing received data.
    """

    def __init__(self,
                 shape: ty.Tuple[int],
                 buffer_size: ty.Optional[int] = 1):
        super().__init__(shape=shape, buffer_size=buffer_size)

        self.buffer = Var(shape=(buffer_size,) + shape, init=0)
        self.in_port = InPort(shape=shape)


@implements(proc=Recv, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvProcModel(PyLoihiProcessModel):
    """Receives dense data from PyInPort and stores it in a Var."""
    buffer: np.ndarray = LavaPyType(np.ndarray, np.float32)
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._buffer_size = proc_params["buffer_size"]

    def run_spk(self) -> None:
        data = self.in_port.recv()
        self.buffer[
            (self.time_step - 1) % self._buffer_size] = data

class TestPropheseeCamera(unittest.TestCase):
    def test_init(self):
        """Test that the PropheseeCamera Process is instantiated correctly."""
        num_output_time_bins = 3

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        camera = PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                                 sensor_shape=(height, width),
                                 num_output_time_bins=num_output_time_bins)

        self.assertIsInstance(camera, PropheseeCamera)

        desired_shape = (num_output_time_bins, 2, height, width)
        self.assertEqual(camera.shape, desired_shape)
        self.assertEqual(camera.s_out.shape, desired_shape)

    def test_invalid_parameters(self):
        """Test that instantiating the PropheseeCamera Process with an invalid
        parameters raises errors."""

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        max_events_per_dt = -12
        num_output_time_bins = -1
        biases = {"bias_diff": 80}

        with self.assertRaises(ValueError):
            PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                            sensor_shape=(height, width),
                            max_events_per_dt=max_events_per_dt)

        with self.assertRaises(ValueError):
            PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                            sensor_shape=(height, width),
                            num_output_time_bins=num_output_time_bins)

        with self.assertRaises(ValueError):
            PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                            sensor_shape=(height, width),
                            biases=biases)


class TestPyPropheseeCameraModel(unittest.TestCase):
    def test_init(self):
        """Test that the PyPropheseeCameraModel ProcessModel is instantiated
        correctly."""

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        transformations = Compose(
            [
                Downsample(factor=0.5),
            ]
        )
        num_output_time_bins = 2

        proc_params = {"shape": (num_output_time_bins, 2, height, width),
                       "device": SEQUENCE_FILENAME_RAW,
                       "biases": None,
                       "filters": [ActivityNoiseFilterAlgorithm(width=width, height=height, threshold=1000)],
                       "max_events_per_dt": 10 ** 8,
                       "transformations": transformations,
                       "num_output_time_bins": num_output_time_bins}

        pm = PyPropheseeCameraModel(proc_params)

        self.assertIsInstance(pm, PyPropheseeCameraModel)
        self.assertIsInstance(pm.reader, RawReader)

    def test_base_functionality_file(self):
        """Test that running a PropheseeCamera works using a data file."""
        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        num_steps = 2

        camera = PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                                 sensor_shape=(height, width))

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    @unittest.skip("Needs live camera")
    def test_base_functionality_camera(self):
        """Test that running a PropheseeCamera works using a camera."""
        num_steps = 2

        camera = PropheseeCamera(device="",
                                 sensor_shape=(720, 1280))

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    @unittest.skip("Needs live camera")
    def test_biases(self):
        """Test that setting biases works"""

        num_steps = 2
        biases = {'bias_diff': 80,
                  'bias_diff_off': 25,
                  'bias_diff_on': 140,
                  'bias_fo': 74,
                  'bias_hpf': 0,
                  'bias_refr': 68, }

        camera = PropheseeCamera(device="",
                                 biases=biases,
                                 sensor_shape=(720, 1280))

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_filters(self):
        """Test that setting biases works"""

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        num_steps = 2
        filters = [ActivityNoiseFilterAlgorithm(width=width, height=height, threshold=1000)]

        camera = PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                                 sensor_shape=(height, width),
                                 filters=filters)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_transformations(self):
        """Test that setting biases works"""

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        num_steps = 2
        transformations = Compose(
            [
                Downsample(factor=0.5),
            ]
        )
        camera = PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                                 sensor_shape=(height, width),
                                 transformations=transformations)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_data_received(self):
        """Test that the PropheseeCamera process correctly reads and sends data."""

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        num_steps = 1

        transformations = Compose(
            [
                Downsample(factor=0.1),
            ]
        )

        camera = PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                                 sensor_shape=(height, width),
                                 transformations=transformations)

        recv = Recv(shape=camera.s_out.shape, buffer_size=num_steps)

        camera.s_out.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg()
        camera.run(condition=run_condition, run_cfg=run_cfg)
        recv_data = recv.buffer.get()
        camera.stop()

        self.assertTrue(np.any(recv_data > 0))

    def test_pause(self):
        """Test pausing the network does not cause any harm. 
        Data will get droped for the pause duration."""

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        del reader

        camera = PropheseeCamera(device=SEQUENCE_FILENAME_RAW,
                                 sensor_shape=(height, width))

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()
        camera.run(condition=run_condition, run_cfg=run_cfg)
        time.sleep(0.1)
        camera.pause()
        time.sleep(0.1)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        time.sleep(0.1)
        camera.stop()

