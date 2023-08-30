# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import sys

try:
    import metavision_core
except ImportError:
    print("Need `metavision` library installed.", file=sys.stderr)
    exit(1)

import numpy as np
import time
from threading import Thread

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.lib.peripherals.dvs.transformation import Compose, EventVolume

from metavision_core.event_io import RawReader, EventDatReader
from metavision_core.event_io import EventsIterator
from metavision_ml.preprocessing.event_to_tensor import histo_quantized


class PropheseeCamera(AbstractProcess):
    """
    Process that receives events from Prophesee device and sends them out as a
    histogram.

    Parameters
    ----------
    filename: str
        String to filename if reading from a RAW/DAT file or empty string for
        using a camera.
    biases: dict
        Dictionary of biases for the DVS Camera.
    filters: list
        List containing metavision filters.
    max_events_per_dt: int
        Maximum events that can be buffered in each timestep.
    transformations: Compose
        Transformations to be applied to the events before sending them out.
    num_output_time_bins: int
        The number of output time bins to use for the ToFrame transformation.
    """

    def __init__(
        self,
        sensor_shape: tuple,
        filename: str = "",
        biases: dict = None,
        filters: list = [],
        max_events_per_dt: int = 10**8,
        transformations: Compose = None,
        num_output_time_bins: int = 1,
        out_shape: tuple = None,
    ):
        if not isinstance(max_events_per_dt, int) or max_events_per_dt < 0:
            raise ValueError(
                "max_events_per_dt must be a positive integer value."
            )

        if (
            not isinstance(num_output_time_bins, int)
            or num_output_time_bins < 0
        ):
            raise ValueError(
                "num_output_time_bins must be a positive integer value."
            )

        if biases is not None and not filename == "":
            raise ValueError("Cant set biases if reading from file.")

        self.filename = filename
        self.biases = biases

        self.max_events_per_dt = max_events_per_dt
        self.filters = filters
        self.transformations = transformations
        self.num_output_time_bins = num_output_time_bins

        height, width = sensor_shape

        if out_shape is not None:
            self.shape = out_shape
        # Automatically determine out_shape
        else:
            event_shape = EventVolume(height=height, width=width, polarities=2)
            if transformations is not None:
                event_shape = self.transformations.determine_output_shape(
                    event_shape
                )
            self.shape = (
                num_output_time_bins,
                event_shape.polarities,
                event_shape.height,
                event_shape.width,
            )

        # Check whether provided transformation is valid
        if self.transformations is not None:
            try:
                # Generate some artificial data
                n_random_spikes = 1000
                test_data = np.zeros(
                    n_random_spikes,
                    dtype=np.dtype(
                        [("y", int), ("x", int), ("p", int), ("t", int)]
                    ),
                )
                test_data["x"] = np.random.rand(n_random_spikes) * width
                test_data["y"] = np.random.rand(n_random_spikes) * height
                test_data["p"] = np.random.rand(n_random_spikes) * 2
                test_data["t"] = np.sort(np.random.rand(n_random_spikes) * 1e6)

                # Transform data
                self.transformations(test_data)
                if len(test_data) > 0:
                    volume = np.zeros(self.shape, dtype=np.uint8)
                    histo_quantized(test_data, volume, np.max(test_data["t"]))

            except Exception:
                raise Exception(
                    "Your transformation is not compatible with the provided \
                    data."
                )

        self.s_out = OutPort(shape=self.shape)

        super().__init__(
            shape=self.shape,
            biases=self.biases,
            filename=self.filename,
            filters=self.filters,
            max_events_per_dt=self.max_events_per_dt,
            transformations=self.transformations,
            num_output_time_bins=self.num_output_time_bins,
        )


class EventsIteratorWrapper():
    """
    PropheseeEventsIterator class for PropheseeCamera which will create a
    thread in the background to always grab events within a time window and
    put them in a buffer.
    
    delta_t: Control the size of the time window for collecting events,
    and generate frames for the events in this time window.Increase delta_t,
    a frame can capture more events, but when you move fast, there will be
    too many events, at this time, you should consider reducing delta_t.
    If delta_t is too large and greater than time interval between two steps
    of processmodel, the thread will recv the old events data from the
    previous time step.

    Parameters
    ----------
    device: str
        String to filename if reading from a RAW/DAT file or empty string for
        using a camera.
    sensor_shape: (int, int)
        Shape of the camera sensor or file recording.
    biases: list
        Bias settings for camera.
    """
    def __init__(self,
                 device: str,
                 sensor_shape: tuple,
                 biases: dict = None,):
        self.true_height, self.true_width = sensor_shape

        self.mv_iterator = EventsIterator(input_path=device, delta_t=1000)

        if biases is not None:
            # Setting Biases for DVS camera
            device_biases = self.mv_iterator.reader.device.get_i_ll_biases()
            for k, v in biases.items():
                device_biases.set(k, v)

        self.thread = Thread(target=self.recv_from_dvs)
        self.stop = False
        self.res = np.zeros(((self.true_width, self.true_height) + (1,) + (1,)),
                            dtype=np.dtype([("y", int), ("x", int),
                                            ("p", int), ("t", int)]))

    def start(self):
        self.thread.start()

    def join(self):
        self.stop = True
        self.thread.join()

    def get_events(self):
        return self.res

    def recv_from_dvs(self):
        for evs in self.mv_iterator:
            self.res = evs

            if self.stop:
                return


@implements(proc=PropheseeCamera, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyPropheseeCameraModel(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params["shape"]
        (
            self.num_output_time_bins,
            self.polarities,
            self.height,
            self.width,
        ) = self.shape
        self.filename = proc_params["filename"]
        self.filters = proc_params['filters']
        self.max_events_per_dt = proc_params['max_events_per_dt']
        self.biases = proc_params['biases']
        self.transformations = proc_params['transformations']
        self.sensor_shape = (self.height,
                             self.width)

        self.reader = EventsIteratorWrapper(
            device=self.filename,
            sensor_shape=self.sensor_shape,
            biases=self.biases)
        self.reader.start()

        self.volume = np.zeros(
            (
                self.num_output_time_bins,
                self.polarities,
                self.height,
                self.width,
            ),
            dtype=np.uint8,
        )

    def run_spk(self):
        """
        Load events from DVS, apply filters and transformations and send
        spikes as frame
        """
        events = self.reader.get_events()

        # Apply filters to events
        for filter in self.filters:
            events_out = filter.get_empty_output_buffer()
            filter.process_events(events, events_out)
            events = events_out

        if len(self.filters) > 0:
            events = events.numpy()

        # Transform events
        if self.transformations is not None and len(events) > 0:
            self.transformations(events)

        # Transform to frame
        if len(events) > 0:
            histo_quantized(events, self.volume, 1000, reset=True)
            frames = self.volume
        else:
            frames = np.zeros(self.s_out.shape)

        self.s_out.send(frames)

    def _pause(self):
        """Pause was called by the runtime"""
        super()._pause()
        self.t_pause = time.time_ns()

    def _stop(self):
        """
        Stop was called by the runtime.
        Helper thread for DVS is also stopped.
        """
        self.reader.join()
        super()._stop()
