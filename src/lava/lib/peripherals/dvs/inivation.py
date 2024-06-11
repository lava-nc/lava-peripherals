# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import time
import threading
from queue import SimpleQueue
import typing as ty
# Inivation camera library
import dv_processing as dv
import inivation_preproc as preproc

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


# TODO: Implement reading from file
class InivationCamera(AbstractProcess):
    """
    Process that receives events from Inivation camera device.

    Parameters:
    -----------
        camera_type: str, optional
            Specify which camera to use. Only used when multiple cameras connected. Defaults to None.
        noise_filter: str, optional
            Type of noise filter to apply to the data. Defaults to None.
        filename: str, optional
            Path to the data if reading from file. Defaults to None.
        flatten: bool, optional
            Whether to flatten the output data array. Defaults to True.
        crop_params: tuple, optional
            A tuple (crop_l, crop_r, crop_t, crop_b) specifying the
            left, right, top, and bottom cropping boundaries.
            Margin definitions of cropping:
            - crop_x_l is the left boundary (inclusive),
            - crop_x_r is the right boundary (exclusive),
            - crop_y_t is the top boundary (inclusive),
            - crop_y_b is the bottom boundary (exclusive).
            Defaults to None.
    Returns:
        None
    """
    def __init__(self,
                camera_type: str = None,
                noise_filter: str = None,
                filename: str = None,
                flatten: bool = False,
                crop_params: list = None,
        ) -> None:

        self.filename = filename
        self.flatten = flatten
        self.__assert_correct_crop_params(crop_params)

        # Setup camera from definition or discover from connected cameras
        if camera_type is None:
            self.camera_type = dv.io.discoverDevices()[0]
        else:
            self.__assert_camera_available(camera_type)
            self.camera_type = camera_type

        # Create the Inivation camera object
        camera = dv.io.CameraCapture(self.camera_type)

        # Setup noise filter for camera
        self.noise_filter = noise_filter

        # Define output shape from initialisation or camera resolution
        self.cam_shape = camera.getEventResolution()

        # If applying cropping, change size of output port
        if crop_params is not None:
            self.crop_params = crop_params

            x_reduction = self.crop_params[0] + self.crop_params[1]
            y_reduction = self.crop_params[2] + self.crop_params[3]

            out_x = self.cam_shape[1] - x_reduction
            out_y = self.cam_shape[0] - y_reduction

            self.out_shape = (out_y, out_x)
        else:
            self.out_shape = self.cam_shape
            self.crop_params = None

        if self.flatten:
            self.out_shape = (np.prod(self.out_shape), )

        self.s_out = OutPort(shape=self.out_shape)

        super().__init__(
            camera_type=self.camera_type,
            cam_shape=self.cam_shape,
            out_shape=self.out_shape,
            noise_filter=self.noise_filter,
            flatten=self.flatten,
            crop_params=self.crop_params,
        )

    def __assert_camera_available(self, camera_type) -> None:
        if camera_type not in dv.io.discoverDevices():
            raise Exception("Camera selected is not detected")

    def __assert_correct_crop_params(self, crop_params) -> None:
        if crop_params is not None:
            if len(crop_params) != 4:
                raise ValueError(
                    f"Crop arguments should be x_l, x_r, y_t, y_b."
                )


class CameraThread():
    """
    This object keeps stores events to the queue in a background thread
    """
    def __init__(self,
                camera_type,
            ) -> None:
        self.camera_type = camera_type
        self.stop = False
        self.sync = False
        self.cam_thread = threading.Thread(
            target=self.store_events, daemon=False
        )
        self.queue = SimpleQueue()

        # Buffer to hold events
        self.events = dv.EventStore()

        # Camera init and config
        self._camera = dv.io.CameraCapture(self.camera_type)
        # Set DAVIS cams to only output events - disables frame capture
        if self._camera.getCameraName().find("DAVIS") != -1:
            self._camera.setDavisReadoutMode(
                dv.io.CameraCapture.DavisReadoutMode.EventsOnly
            )

    # Return the starting time of data gathering in milliseconds
    def get_starttime(self) -> None:
        return self._starttime

    def start(self) -> None:
        self.cam_thread.start()

    def join(self) -> None:
        self.stop = True
        self.cam_thread.join()

    def get_events(self):
        return self.queue.get()

    def store_events(self) -> None:
        while self.stop is not True:
            # Get batch of events from camera
            # NOTE: If no sync flag then this continually drains the cam buffer
            batch = self._camera.getNextEventBatch()

            if self.sync is True:
                # Add events to the event queue to be read from main proc
                if batch is not None:
                    self.queue.put(batch)

                # NOTE: This timing delay is key for operating in "real" time
                time.sleep(0.001)

    def set_start_time(self, start_time) -> None:
        self.start_time = start_time

    def empty_buffer(self) -> None:
        # Flag to start addding to queue
        self.sync = True


@implements(proc=InivationCamera, protocol=LoihiProtocol)
@requires(CPU)
class PySparseInivationCameraModel(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.camera_type = proc_params["camera_type"]
        self.cam_shape = proc_params["cam_shape"]
        self.out_shape = proc_params["out_shape"]
        self.noise_filter = proc_params["noise_filter"]
        self.flatten = proc_params["flatten"]
        self.crop_params = proc_params["crop_params"]
        self.start_time = None

        self.reader = CameraThread(
            camera_type = self.camera_type,
        )
        # Start background dvs camera thread
        self.reader.start()

    def run_spk(self) -> None:
        # On first iteration clear the cameras buffer to ensure time sync
        if self.time_step == 1:
            self.reader.empty_buffer()

        # Take event batch from buffer
        self.current_batch = self.reader.get_events()

        if self.current_batch is not None:
            if self.noise_filter is not None:
                self.noise_filter.accept(self.current_batch)
                self.current_batch = self.noise_filter.generateEvents()

            data, indices = self._create_sparse_vector(self.current_batch)

            # Apply any preprocessing steps
            if self.crop_params is not None:
                data, indices = preproc.crop(data,
                                            indices,
                                            self.cam_shape,
                                            self.crop_params,
                                            self.out_shape
                                        )
        else:
            data = np.zeros(self.out_shape)
            indices = np.zeros(self.out_shape)

        # Output spikes
        self.s_out.send(data, indices)

    def _create_sparse_vector(self, evs) -> ty.Tuple[np.ndarray, np.ndarray]:
        """ Create sparse vector from an event batch"""
        data = evs.polarities()
        coords = evs.coordinates()
        x = coords[:, 0]
        y = coords[:, 1]

        indices = np.ravel_multi_index((x, y), self.cam_shape)

        return data, indices

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """
        Stop was called by the runtime.
        Helper thread for DVS is also stopped.
        """
        self.reader.join()
        super()._stop()
