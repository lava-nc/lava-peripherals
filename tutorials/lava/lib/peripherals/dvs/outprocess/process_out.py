# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class ProcessOut(AbstractProcess):
    """
    Process that receives frame as well as direction neuron input, computes
    the direction vector and sends it out via a Mulitprocessing Pipe.

    Parameters
    ----------
    frame_shape: tuple, shape of the input frame
    direction_shape: tuple, shape of direction neurons
    send_pipe: Pipe, pipe through which data should be sent
    """

    def __init__(self, frame_shape, direction_shape, send_pipe):
        super().__init__(frame_shape=frame_shape,
                         direction_shape=direction_shape,
                         send_pipe=send_pipe)
        self.frame_shape = frame_shape
        self.direction_shape = direction_shape
        self.frame_in = InPort(shape=frame_shape)
        self.direction_in_left = InPort(shape=direction_shape)
        self.direction_in_right = InPort(shape=direction_shape)


@implements(proc=ProcessOut, protocol=LoihiProtocol)
@requires(CPU)
class ProcessOutModel(PyLoihiProcessModel):
    frame_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    direction_in_left: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    direction_in_right: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.frame_shape = proc_params["frame_shape"]
        self.direction_shape = proc_params["direction_shape"]
        self._send_pipe = proc_params["send_pipe"]
        self.height = self.frame_shape[2]
        self.width = self.frame_shape[3]


    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        frame = self.frame_in.recv()
        left = self.direction_in_left.recv().sum()
        right = self.direction_in_right.recv().sum()

        frame = np.rot90(frame.sum(axis=0).sum(axis=0))
        frame = np.rot90(frame)
        lr_arrow_start = (self.width // 2, self.height // 2)
        lr_arrow_end = (
            self.width // 2 + int(left-right) * 3,
            self.height // 2,
        )
        lr_arrow_end = np.clip(lr_arrow_end, (0, 0), (self.width, self.height))
        data_dict = {
            "dvs_frame": frame,
            "arrow": [lr_arrow_start, lr_arrow_end]
        }
        self._send_pipe.send(data_dict)
