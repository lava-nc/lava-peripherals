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
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, shape, direction_shape, send_pipe):
        super().__init__(shape=shape,
                         direction_shape=direction_shape,
                         send_pipe=send_pipe)
        self.shape = shape
        self.direction_shape = direction_shape
        self.frame_in = InPort(shape=shape)
        self.direction_in = InPort(shape=direction_shape)


@implements(proc=ProcessOut, protocol=LoihiProtocol)
@requires(CPU)
class ProcessOutModel(PyLoihiProcessModel):
    frame_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    direction_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params["shape"]
        self.direction_shape = proc_params["direction_shape"]
        self._send_pipe = proc_params["send_pipe"]
        self.height = self.shape[2]
        self.width = self.shape[3]



    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        frame = self.frame_in.recv()
        direction_in = self.direction_in.recv()
        n_per_direction = self.direction_shape[0] // 2

        left = direction_in[:n_per_direction].sum()
        right = direction_in[n_per_direction:2 * n_per_direction].sum()


        frame = np.rot90(frame.sum(axis=0).sum(axis=0))
        frame = np.rot90(frame)

        lr_arrow_start = (self.width // 2, self.height // 2)
        lr_arrow_end = (
            self.width // 2 + int(right - left) * 3,
            self.height // 2,
        )
        lr_arrow_end = np.clip(lr_arrow_end, (0, 0), (self.width, self.height))

        data_dict = {
            "dvs_frame": frame,
            "arrow": [lr_arrow_start, lr_arrow_end]
        }
        self._send_pipe.send(data_dict)
