# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import sys

try:
    from metavision_core.event_io import (EventsIterator, RawReader,
                                          EventDatReader)
    from metavision_ml.preprocessing.event_to_tensor import histo_quantized
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
    mode: str
        String to specify to load events by numbers of events, timeslice or
        the first met criterion. (Choice of "mixed", "delta_t", or "n_events").
        Default is "mixed".
    n_events: int
        Number of events in the timeslice.
    delta_t: int
        Duration of served event slice in us.
    transformations: Compose
        Transformations to be applied to the events before sending them out.
    num_output_time_bins: int
        The number of output time bins to use for the ToFrame transformation.
    sync_time: bool
        Whether to synchronize camera timestamps with Loihi 2 runtime.
    flatten: bool
        Whether to send flattened output.
    """

    def __init__(
            self,
            sensor_shape: tuple,
            filename: str = "",
            biases: dict = None,
            filters: list = [],
            mode: str = "mixed",
            n_events: int = 10 ** 8,
            delta_t: int = 1000,
            transformations: Compose = None,
            num_output_time_bins: int = 1,
            out_shape: tuple = None,
            sync_time: bool = True,
            flatten: bool = False
    ):
        if not isinstance(n_events, int) or n_events < 0:
            raise ValueError(
                "n_events must be a positive integer value."
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
        self.sync_time = sync_time
        self.flatten = flatten
        self.mode = mode
        self.n_events = n_events
        self.delta_t = delta_t
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
        num_neurons = np.prod(self.shape)
        if self.flatten:
            self.s_out = OutPort(shape=(num_neurons,))
        else:
            self.s_out = OutPort(shape=self.shape)

        super().__init__(
            shape=self.shape,
            biases=self.biases,
            filename=self.filename,
            filters=self.filters,
            mode=self.mode,
            n_events=self.n_events,
            delta_t=self.delta_t,
            transformations=self.transformations,
            num_output_time_bins=self.num_output_time_bins,
            sync_time=self.sync_time,
            flatten=self.flatten
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
    mode: str
        String to specify to load events by numbers of events, timeslice or
        the first met criterion. (Choice of "mixed", "delta_t", or "n_events").
        Default is "mixed".
    n_events: int
        Number of events in the timeslice.
    delta_t: int
        Duration of served event slice in us.
    biases: list
        Bias settings for camera.
    """

    def __init__(self,
                 device: str,
                 sensor_shape: tuple,
                 mode: str = "mixed",
                 n_events: int = 1000,
                 delta_t: int = 1000,
                 biases: dict = None, ):
        self.true_height, self.true_width = sensor_shape

        self.mv_iterator = EventsIterator(input_path=device, mode=mode,
                                          n_events=n_events, delta_t=delta_t)

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
@tag("floating_pt", "event_it")
class PyPropheseeCameraEventsIteratorModel(PyLoihiProcessModel):
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
        self.mode = proc_params['mode']
        self.n_events = proc_params['n_events']
        self.delta_t = proc_params['delta_t']
        self.biases = proc_params['biases']
        self.transformations = proc_params['transformations']
        self.sensor_shape = (self.height,
                             self.width)

        self.reader = EventsIteratorWrapper(
            device=self.filename,
            sensor_shape=self.sensor_shape,
            mode=self.mode,
            n_events=self.n_events,
            delta_t=self.delta_t,
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


@implements(proc=PropheseeCamera, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt", "raw_reader")
class PyPropheseeCameraRawReaderModel(PyLoihiProcessModel):
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
        self.filters = proc_params["filters"]
        self.n_events = proc_params["n_events"]
        self.biases = proc_params["biases"]
        self.transformations = proc_params["transformations"]
        self.flatten = proc_params["flatten"]
        self.sync_time = proc_params["sync_time"]

        if self.filename.split('.')[-1] == 'dat':
            self.reader = EventDatReader(self.filename)
        else:
            self.reader = RawReader(self.filename,
                                    max_events=self.n_events)

        if self.biases is not None:
            # Setting Biases for DVS camera
            device_biases = self.reader.device.get_i_ll_biases()
            for k, v in self.biases.items():
                device_biases.set(k, v)

        self.volume = np.zeros(
            (
                self.num_output_time_bins,
                self.polarities,
                self.height,
                self.width,
            ),
            dtype=np.uint8,
        )
        self.t_pause = None
        self.t_last_iteration = None

    def run_spk(self):
        """Load events from DVS, apply filters and transformations and send
        spikes as frame"""

        if self.sync_time:
            if self.t_pause is None:
                self.t_pause = time.time_ns()

            if self.t_last_iteration is None:
                self.t_last_iteration = time.time_ns()

            # Time passed since last iteration
            t_now = time.time_ns()

            # Load new events since last iteration
            if self.t_pause > self.t_last_iteration:
                # Runtime was paused in the meantime
                delta_t = np.max(
                    [10000, (self.t_pause - self.t_last_iteration) // 1000]
                )
                delta_t_drop = np.max([10000, (t_now - self.t_pause) // 1000])

                events = self.reader.load_delta_t(delta_t)
                _ = self.reader.load_delta_t(delta_t_drop)
            else:
                # Runtime was not paused in the meantime
                delta_t = np.max([10000,
                                  (t_now - self.t_last_iteration) // 1000])
                events = self.reader.load_delta_t(delta_t)
        else:
            delta_t = 10000
            events = self.reader.load_delta_t(delta_t)

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
            histo_quantized(events, self.volume, delta_t, reset=True)
            frames = self.volume
        else:
            frames = np.zeros(self.s_out.shape)

        # Send
        if self.flatten:
            frames = frames.flatten()
        self.s_out.send(frames)
        self.t_last_iteration = time.time_ns()

    def _pause(self):
        """Pause was called by the runtime"""
        super()._pause()
        self.t_pause = time.time_ns()
