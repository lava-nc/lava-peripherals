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
from enum import Enum
import typing as ty

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel, PyAsyncProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
import math  
from tonic import transforms
import inspect
from metavision_core.event_io import RawReader
from metavision_ml.preprocessing.event_to_tensor import histo_quantized

from tonic.transforms import Compose
import tonic
import warnings

class PropheseeCamera(AbstractProcess):
    """
    Process that receives events from Prophesee device and sends them out as a histogram. 

    Parameters
    ----------
    device: str
        String to filename if reading from a RAW file or empty string for using a camera.
    biases: dict
        Dictionary of biases for the DVS Camera.
    filters: list
        List containing metavision filters. 
    max_events_per_dt: int
        Maximum events that can be buffered in each timestep.
    transformations: Compose
        Tonic transformations to be applied to the events before sending them out.
    num_output_time_bins: int
        The number of output time bins to use for the ToFrame transformation.
    """

    def __init__(self,
                 sensor_shape: tuple,
                 device: str,
                 biases: dict = None,
                 filters: list = [],
                 max_events_per_dt: int = 10 ** 8,
                 transformations: Compose = None,
                 num_output_time_bins: int = 1,
                 out_shape: tuple = None,
                 ):

        if not isinstance(max_events_per_dt, int) or max_events_per_dt < 0:
            raise ValueError("max_events_per_dt must be a positive integer value.")

        if not isinstance(num_output_time_bins, int) or num_output_time_bins < 0:
            raise ValueError("num_output_time_bins must be a positive integer value.")

        if not biases is None and not device == "":
            raise ValueError("Cant set biases if reading from file.")

        self.device = device
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
            polarities, height, width = self._determine_output_shape(2, height, width, transformations)
            self.shape = (num_output_time_bins, polarities, height, width)

        # Check whether provided transformation is valid
        if self.transformations is not None:
            try:
                # Generate some artificial data
                n_random_spikes = 1000 
                test_data = np.zeros(n_random_spikes, dtype=np.dtype([("y", int), ("x", int), ("p", int), ("t", int)]))
                test_data["x"] = np.random.rand(n_random_spikes) * width
                test_data["y"] = np.random.rand(n_random_spikes) * height
                test_data["p"] = np.random.rand(n_random_spikes) * 2
                test_data["t"] = np.sort(np.random.rand(n_random_spikes) * 1e6)

                # Transform data
                test_data = self.transformations(test_data)
                if len(test_data) > 0:
                    volume = np.zeros(self.shape, dtype=np.uint8)
                    histo_quantized(test_data, volume, np.max(test_data['t']))

            except Exception:
                raise Exception("Your transformer is not compatible with the provided data.")

        self.s_out = OutPort(shape=self.shape)

        super().__init__(shape=self.shape,
                         biases=self.biases,
                         device=self.device,
                         filters=self.filters,
                         max_events_per_dt=self.max_events_per_dt,
                         transformations=self.transformations,
                         num_output_time_bins=self.num_output_time_bins)

    def _determine_output_shape(self, polarities, height, width, transformations):
        invalid_transforms = {tonic.transforms.ToVoxelGrid,
                              tonic.transforms.ToImage,
                              tonic.transforms.ToBinaRep,
                              tonic.transforms.ToTimesurface,
                              tonic.transforms.ToAveragedTimesurface,
                              tonic.transforms.ToSparseTensor,
                              tonic.transforms.ToOneHotEncoding,
                              tonic.transforms.ToFrame}

        if transformations is not None:
            for transform in transformations.transforms:

                # Invalid Transforms
                if type(transform) in invalid_transforms:
                    raise TypeError(f"You can only use event transformations, "
                                    f"Compose object contained event representation {transform}.")

                # Shape Changing Transforms
                elif type(transform) == tonic.transforms.MergePolarities:
                    polarities = 1
                elif type(transform) == tonic.transforms.Downsample:
                    height = math.ceil(height * transform.spatial_factor)
                    width = math.ceil(width * transform.spatial_factor)
                elif type(transform) == tonic.transforms.CenterCrop:
                    if type(transform.size) is int:
                        height = width = transform.size
                    else:
                        height, width, = transform.size
                elif type(transform) == tonic.transforms.RandomCrop:
                    height, width = transform.target_size

                # Composition
                elif type(transform) == tonic.transforms.Compose:
                    polarities, height, width = self._determine_output_shape(polarities, height, width, transform)

                # Otherwise, it should be in the non shape changing 
                else:
                    _, tonic_transforms = zip(*inspect.getmembers(transforms))
                    if not (type(transform) in tonic_transforms and inspect.isclass(type(transform))):
                        warnings.warn(f"Unknown transformation {transform}."
                                      f"Automatic shape detection may be incorrect.")
        return (polarities, height, width)


@implements(proc=PropheseeCamera, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyPropheseeCameraModel(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params['shape']
        self.num_output_time_bins, _, self.height, self.width = self.shape
        self.device = proc_params['device']
        self.filters = proc_params['filters']
        self.max_events_per_dt = proc_params['max_events_per_dt']
        self.biases = proc_params['biases']
        self.transformations = proc_params['transformations']

        self.reader = RawReader(self.device, max_events=self.max_events_per_dt)

        if not self.biases is None:
            # Setting Biases for DVS camera
            device_biases = self.reader.device.get_i_ll_biases()
            for k, v in self.biases.items():
                device_biases.set(k, v)

     
        self.volume = np.zeros((self.num_output_time_bins, 2, self.height, self.width), dtype=np.uint8)
        self.t_pause = time.time_ns()
        self.t_last_iteration = time.time_ns()

    def run_spk(self):
        """Load events from DVS, apply filters and transformations and send spikes as frame """

        # Time passed since last iteration
        t_now = time.time_ns()

        # Load new events since last iteration
        if self.t_pause > self.t_last_iteration:
            # Runtime was paused in the meantime
            delta_t = np.max([10000, (self.t_pause - self.t_last_iteration) // 1000])
            delta_t_drop = np.max([10000, (t_now - self.t_pause) // 1000])

            events = self.reader.load_delta_t(delta_t)
            _ = self.reader.load_delta_t(delta_t_drop)
        else:
            # Runtime was not paused in the meantime
            delta_t = np.max([10000, (t_now - self.t_last_iteration) // 1000])
            events = self.reader.load_delta_t(delta_t)        


        # Apply filters to events
        for filter in self.filters:
            events_out = filter.get_empty_output_buffer()
            filter.process_events(events, events_out)
            events = events_out.numpy()

        # Transform events
        if not self.transformations is None:
            events = self.transformations(events)

        # Transform to frame
        if len(events) > 0:
            histo_quantized(events, self.volume, delta_t)
            frames = self.volume
        else:
            frames = np.zeros(self.s_out.shape)

        # Send 
        self.s_out.send(frames)
        self.t_last_iteration = t_now

    def _pause(self):
        """ Pause was called by the runtime """
        super()._pause()
        self.t_pause = time.time_ns()

