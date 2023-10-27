# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import os
import time
import numpy as np
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
from lava.lib.peripherals.dvs.transformation import Compose, Downsample
from lava.lib.peripherals.dvs.prophesee import (
    PropheseeCamera,
    PyPropheseeCameraRawReaderModel,
    PyPropheseeCameraEventsIteratorModel,
    EventsIteratorWrapper)

from metavision_core.utils import get_sample
from metavision_core.event_io import RawReader, EventsIterator
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm


SEQUENCE_FILENAME_RAW = "sparklers.raw"
get_sample(SEQUENCE_FILENAME_RAW)
assert os.path.isfile(SEQUENCE_FILENAME_RAW)

SEQUENCE_FILENAME_DAT = "blinking_leds_td.dat"
get_sample(SEQUENCE_FILENAME_DAT)
assert os.path.isfile(SEQUENCE_FILENAME_DAT)

# Test if camera is connected
try:
    reader = EventsIterator("", delta_t=1)
    del reader
    USE_CAMERA_TESTS = True
except OSError:
    USE_CAMERA_TESTS = False


def get_shape(file_name):
    mv_iterator = EventsIterator(input_path=file_name,
                                 delta_t=1000)
    height, width = mv_iterator.get_size()
    del mv_iterator
    return height, width


class Recv(AbstractProcess):
    """Process that receives arbitrary dense data and stores it in a Var.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Var.
    buffer_size: optional, int
        Size of buffer storing received data.
    """

    def __init__(self, shape: ty.Tuple[int], buffer_size: ty.Optional[int] = 1):
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
        self.buffer[(self.time_step - 1) % self._buffer_size] = data


class TestPropheseeCamera(unittest.TestCase):
    def test_init(self):
        """Test that the PropheseeCamera Process is
        instantiated correctly."""
        num_output_time_bins = 3

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW,
            sensor_shape=(height, width),
            num_output_time_bins=num_output_time_bins,
        )

        self.assertIsInstance(camera, PropheseeCamera)

        desired_shape = (num_output_time_bins, 2, height, width)
        self.assertEqual(camera.shape, desired_shape)
        self.assertEqual(camera.s_out.shape, desired_shape)

    def test_invalid_parameters(self):
        """Test that instantiating the PropheseeCamera Process with an invalid
        parameters raises errors."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        n_events = -12
        num_output_time_bins = -1
        biases = {"bias_diff": 80}

        with self.assertRaises(ValueError):
            PropheseeCamera(
                filename=SEQUENCE_FILENAME_RAW,
                sensor_shape=(height, width),
                n_events=n_events,
            )

        with self.assertRaises(ValueError):
            PropheseeCamera(
                filename=SEQUENCE_FILENAME_RAW,
                sensor_shape=(height, width),
                num_output_time_bins=num_output_time_bins,
            )

        with self.assertRaises(ValueError):
            PropheseeCamera(
                filename=SEQUENCE_FILENAME_RAW,
                sensor_shape=(height, width),
                biases=biases,
            )


class TestPyPropheseeCameraModel_EventsIt(unittest.TestCase):
    def test_init(self):
        """Test that the PyPropheseeCameraEventsIteratorModel ProcessModel
        is instantiated correctly."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        transformations = Compose(
            [
                Downsample(factor=0.5),
            ]
        )
        num_output_time_bins = 2

        proc_params = {
            "shape": (num_output_time_bins, 2, height, width),
            "filename": SEQUENCE_FILENAME_RAW,
            "biases": None,
            "filters": [
                ActivityNoiseFilterAlgorithm(
                    width=width, height=height, threshold=1000
                )
            ],
            "mode": "mixed",
            "n_events": 10**8,
            "delta_t": 1000,
            "transformations": transformations
        }

        pm = PyPropheseeCameraEventsIteratorModel(proc_params)

        self.assertIsInstance(pm, PyPropheseeCameraEventsIteratorModel)
        self.assertIsInstance(pm.reader, EventsIteratorWrapper)

    def test_base_functionality_file(self):
        """Test that running a PropheseeCamera works using a data file."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        num_steps = 2

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW, sensor_shape=(height, width)
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_base_functionality_dat_file(self):
        """Test that running a PropheseeCamera works
        using a dat data file."""
        # The DAT file should have the same resolution as the RAW file
        height, width = get_shape(SEQUENCE_FILENAME_RAW)
        num_steps = 2

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_DAT,
            sensor_shape=(height, width),
            n_events=1000
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    @unittest.skipUnless(USE_CAMERA_TESTS, "Needs live camera")
    def test_base_functionality_camera(self):
        """Test that running a PropheseeCamera works using a camera."""
        num_steps = 2

        camera = PropheseeCamera(filename="", sensor_shape=(720, 1280))

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    @unittest.skipUnless(USE_CAMERA_TESTS, "Needs live camera")
    def test_biases(self):
        """Test that setting biases works."""
        num_steps = 2
        biases = {
            "bias_diff": 80,
            "bias_diff_off": 25,
            "bias_diff_on": 140,
            "bias_fo": 74,
            "bias_hpf": 0,
            "bias_refr": 68,
        }

        camera = PropheseeCamera(
            filename="", biases=biases, sensor_shape=(720, 1280)
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_filters(self):
        """Test that filters work."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        num_steps = 2
        filters = [
            ActivityNoiseFilterAlgorithm(
                width=width, height=height, threshold=1000
            )
        ]

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW,
            sensor_shape=(height, width),
            filters=filters,
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_transformations(self):
        """Test that transformations work."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        num_steps = 2
        transformations = Compose(
            [
                Downsample(factor=0.5),
            ]
        )
        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW,
            sensor_shape=(height, width),
            transformations=transformations,
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_data_received(self):
        """Test that the PropheseeCamera process correctly reads and sends
        data."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)
        num_steps = 1

        transformations = Compose(
            [
                Downsample(factor=0.1),
            ]
        )

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW,
            sensor_shape=(height, width),
            transformations=transformations,
        )

        recv = Recv(shape=camera.s_out.shape, buffer_size=num_steps)

        camera.s_out.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        recv_data = recv.buffer.get()
        camera.stop()

        self.assertTrue(np.any(recv_data > 0))

    def test_pause(self):
        """Test pausing the network does not cause any harm.
        Data will get droped for the pause duration."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW, sensor_shape=(height, width)
        )

        run_condition = RunContinuous()
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = \
            PyPropheseeCameraEventsIteratorModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        time.sleep(0.1)
        camera.pause()
        time.sleep(0.1)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        time.sleep(0.1)
        camera.stop()


class TestPyPropheseeCameraModel_RawReader(unittest.TestCase):
    def test_init(self):
        """Test that the PyPropheseeCameraRawReaderModel ProcessModel is instantiated
        correctly."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        transformations = Compose(
            [
                Downsample(factor=0.5),
            ]
        )
        num_output_time_bins = 2

        proc_params = {
            "shape": (num_output_time_bins, 2, height, width),
            "filename": SEQUENCE_FILENAME_RAW,
            "biases": None,
            "filters": [
                ActivityNoiseFilterAlgorithm(
                    width=width, height=height, threshold=1000
                )
            ],
            "n_events": 10**8,
            "transformations": transformations,
            "sync_time": True,
            "flatten": False
        }

        pm = PyPropheseeCameraRawReaderModel(proc_params)

        self.assertIsInstance(pm, PyPropheseeCameraRawReaderModel)
        self.assertIsInstance(pm.reader, RawReader)

    def test_base_functionality_file(self):
        """Test that running a PropheseeCamera works using a data file."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        num_steps = 2

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW, sensor_shape=(height, width)
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_base_functionality_dat_file(self):
        """Test that running a PropheseeCamera works
        using a dat data file."""
        # The DAT file should have the same resolution as the RAW file

        height, width = get_shape(SEQUENCE_FILENAME_DAT)
        num_steps = 2

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_DAT, sensor_shape=(height, width)
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    @unittest.skipUnless(USE_CAMERA_TESTS, "Needs live camera")
    def test_base_functionality_camera(self):
        """Test that running a PropheseeCamera works using a camera."""
        num_steps = 2

        camera = PropheseeCamera(filename="", sensor_shape=(720, 1280))

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    @unittest.skipUnless(USE_CAMERA_TESTS, "Needs live camera")
    def test_biases(self):
        """Test that setting biases works."""

        num_steps = 2
        biases = {
            "bias_diff": 80,
            "bias_diff_off": 25,
            "bias_diff_on": 140,
            "bias_fo": 74,
            "bias_hpf": 0,
            "bias_refr": 68,
        }

        camera = PropheseeCamera(
            filename="", biases=biases, sensor_shape=(720, 1280)
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_filters(self):
        """Test that filters work."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        num_steps = 2
        filters = [
            ActivityNoiseFilterAlgorithm(
                width=width, height=height, threshold=1000
            )
        ]

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW,
            sensor_shape=(height, width),
            filters=filters,
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_transformations(self):
        """Test that transformations work."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        num_steps = 2
        transformations = Compose(
            [
                Downsample(factor=0.5),
            ]
        )
        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW,
            sensor_shape=(height, width),
            transformations=transformations,
        )

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        camera.stop()

    def test_data_received(self):
        """Test that the PropheseeCamera process correctly reads and sends
        data."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)
        num_steps = 1

        transformations = Compose(
            [
                Downsample(factor=0.1),
            ]
        )

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW,
            sensor_shape=(height, width),
            transformations=transformations,
        )

        recv = Recv(shape=camera.s_out.shape, buffer_size=num_steps)

        camera.s_out.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        recv_data = recv.buffer.get()
        camera.stop()

        self.assertTrue(np.any(recv_data > 0))

    def test_pause(self):
        """Test pausing the network does not cause any harm.
        Data will get droped for the pause duration."""

        height, width = get_shape(SEQUENCE_FILENAME_RAW)

        camera = PropheseeCamera(
            filename=SEQUENCE_FILENAME_RAW, sensor_shape=(height, width)
        )

        run_condition = RunContinuous()
        custom_proc_model_map = {}
        custom_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel
        run_cfg = Loihi2SimCfg(exception_proc_model_map=custom_proc_model_map)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        time.sleep(0.1)
        camera.pause()
        time.sleep(0.1)
        camera.run(condition=run_condition, run_cfg=run_cfg)
        time.sleep(0.1)
        camera.stop()


if __name__ == "__main__":
    unittest.main()
