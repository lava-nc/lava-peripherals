# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
from multiprocessing import Pipe
from lava.magma.compiler.compiler import Compiler
from lava.magma.compiler.executable import Executable
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.runtime.runtime import Runtime
from lava.lib.peripherals.dvs.prophesee import PropheseeCamera, PyPropheseeCameraRawReaderModel
from lava.lib.peripherals.dvs.transformation import Compose, MergePolarities, \
    Downsample
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
from lava.proc.dense.models import PyDenseModelBitAcc
from lava.proc.lif.models import PyLifModelBitAcc
from lava.proc.sparse.models import PySparseModelBitAcc
from lava.proc.embedded_io.spike import NxToPyAdapter, PyToNxAdapter
from lava.utils.serialization import save
from outprocess.process_out import ProcessOut
from scipy.sparse import csr_matrix
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions

CompilerOptions.verbose = True

prophesee_input_default_config = {
    "filename": "hand_swipe.dat",
    "transformations": Compose(
        [
            Downsample(factor=0.125),
            MergePolarities(),
        ]
    ),
    "sensor_shape": (480, 640),
    "num_output_time_bins": 1,
    "sync_time": False,
    "flatten": True
}

lif_default_config = {
    "vth": 600,
    "du": 2000,
    "dv": 2000
}

weights_default_config = {
    "w_ff": 255,
    "w_rec": 5,
    "w_o": 16,
    "kernel_size": 20
}


class SwipeDetector:
    """
    Class to setup the swipe detection network, compile it, and initialize
    its runtime.

    Parameters
    ----------
    send_pipe: Pipe
        Pipe through which to send output results from Lava to vanilla Python.
    num_steps: int
        Number of time steps the network should be run.
    use_loihi2: bool
        Whether to run on Loihi 2 chip or in simulation.
    prophesee_input_config: dict, optional
        Dictionary containing kwargs for the PropheseeCameraProcess. If none
        is provided the default configuration from above is used.
    lif_config: dict, optional
        Dictionary containing kwargs for the LIF Process. If none
        is provided the default configuration from above is used.
    weights_config: dict, optional
        Dictionary containing weights for connecting Processes.If none
        is provided the default configuration from above is used.
    num_out_neurons: int, optional
        Number of output neurons to collect direction information.
        Default is 50.
    blocking: bool, optional
        Whether the network should run in blocking or non-blocking mode.
        Default is non-blocking.
    executable: Executable, optional
        Which executable to use. In case no executable is provided the net-
        work will be compiled from scratch.
    path_to_save_network: str, optional
        Where to store the (generated) executable. If no path is provided the
        executable will not be saved.
    """

    def __init__(self,
                 send_pipe: type(Pipe),
                 num_steps: int,
                 use_loihi2: bool,
                 prophesee_input_config: ty.Optional[dict] = None,
                 lif_config: ty.Optional[dict] = None,
                 weights_config: ty.Optional[dict] = None,
                 num_out_neurons: ty.Optional[int] = 50,
                 blocking: ty.Optional[bool] = False,
                 executable: ty.Optional[Executable] = None,
                 path_to_save_network: ty.Optional[str, None] = None
                 ) -> None:
        self.prophesee_input_config = prophesee_input_config or \
            prophesee_input_default_config
        self.lif_config = lif_config or lif_default_config
        self.weights_config = weights_config or weights_default_config
        self.send_pipe = send_pipe
        self.num_steps = num_steps
        self.use_loihi2 = use_loihi2
        self.executable = executable
        self.path_to_save_network = path_to_save_network
        self._create_processes()
        self._make_connections()
        print("network created")

        # Run
        if self.use_loihi2:
            print("running on chip")
            run_cfg = Loihi2HwCfg(exception_proc_model_map=
                                  {PropheseeCamera: PyPropheseeCameraRawReaderModel})
        else:
            run_cfg = Loihi2SimCfg(
                exception_proc_model_map={Dense: PyDenseModelBitAcc,
                                          Sparse: PySparseModelBitAcc,
                                          LIF: PyLifModelBitAcc,
                                          PropheseeCamera: PyPropheseeCameraRawReaderModel})

        # Compilation
        if self.executable is None:
            compiler = Compiler()
            self.executable = compiler.compile(self.frame_input, run_cfg=run_cfg)

        if self.path_to_save_network is not None:
            self._store_network_executable()


        # Initialize runtime
        mp = ActorType.MultiProcessing
        self.runtime = Runtime(exe=executable,
                               message_infrastructure_type=mp)
        self.runtime.initialize()

    def _store_network_executable(self):
        # Store the Lava network, only needed if network changes
        save(processes=[self.ff_inp,
                        self.ff_left,
                        self.rec_left,
                        self.lif_left,
                        self.ff_right,
                        self.rec_right,
                        self.lif_right,
                        self.sparse_out_left,
                        self.sparse_out_right,
                        self.sparse_out_left_inv,
                        self.sparse_out_right_inv,
                        self.out_lif_left,
                        self.out_lif_right],
             filename= self.path_to_save_network,
             executable=self.executable)


    def _create_processes(self) -> None:
        # Create Processes and Weights
        self.frame_input = PropheseeCamera(**self.prophesee_input_config)
        self.num_neurons = np.prod(self.frame_input.s_out.shape)
        _, _, self.scaled_height, self.scaled_width = self.frame_input.shape
        self.flat_shape = (self.num_neurons,)

        # Create weights
        # FF weights
        self.ff_weights = np.eye(self.num_neurons) * self.weights_config["w_ff"]

        # Left weights
        self.rec_weights_left = np.zeros((self.num_neurons, self.num_neurons))
        self.rec_weights_left = self.rec_weights_left.reshape(
            (self.scaled_height, self.scaled_width,
             self.scaled_height, self.scaled_width))

        for i in range(self.scaled_height):
            for j in range(self.scaled_width):
                ks = self.weights_config["kernel_size"]
                self.rec_weights_left[i, j - ks :j, i, j] = \
                    self.weights_config["w_rec"]

        self.rec_weights_left = csr_matrix(
            self.rec_weights_left.reshape((self.num_neurons, self.num_neurons)))

        # Right weights
        self.rec_weights_right = np.zeros((self.num_neurons, self.num_neurons))
        self.rec_weights_right = self.rec_weights_right.reshape(
            (self.scaled_height, self.scaled_width,
             self.scaled_height, self.scaled_width))
        for i in range(self.scaled_height):
            for j in range(self.scaled_width):
                ks = self.weights_config["kernel_size"]
                self.rec_weights_right[i, j + 1:j + ks + 1, i, j] = \
                    self.weights_config["w_rec"]

        self.rec_weights_right = csr_matrix(
            self.rec_weights_right.reshape(
                (self.num_neurons, self.num_neurons)))

        # Weights out
        self.w_out_left = np.zeros((self.num_out_neurons, self.num_neurons))
        self.w_out_right = np.zeros((self.num_out_neurons, self.num_neurons))

        tmp = np.kron(np.eye(self.num_out_neurons, dtype=np.int32),
                      np.array([self.weights_config["w_o"]]
                               * (self.num_neurons // self.num_out_neurons)))

        self.w_out_left[:self.num_out_neurons, :tmp.shape[1]] = tmp
        self.w_out_right[:self.num_out_neurons, :tmp.shape[1]] = tmp

        self.w_out_left[:, 0] = 1  # bug fix
        self.w_out_right[:, 0] = 1  # bug fix

        # Create in processes
        self.ff_inp = Sparse(weights=self.ff_weights, num_message_bits=8)
        self.lif_inp = LIF(shape=self.frame_input.shape, **self.lif_config)

        # Create left processes
        self.ff_left = Sparse(weights=self.ff_weights)
        self.rec_left = Sparse(weights=self.rec_weights_left)
        self.lif_left = LIF(shape=self.frame_input.shape, **self.lif_config)

        #Create right processes
        self.ff_right = Sparse(weights=self.ff_weights)
        self.rec_right = Sparse(weights=self.rec_weights_right)
        self.lif_right = LIF(shape=self.frame_input.shape, **self.lif_config)

        #Create out processes
        self.sparse_out_left = Sparse(weights=self.w_out_left)
        self.sparse_out_right = Sparse(weights=self.w_out_right)
        self.sparse_out_left_inv = Sparse(weights=-self.w_out_right)
        self.sparse_out_right_inv = Sparse(weights=-self.w_out_left)
        self.out_lif_left = LIF(shape=(self.num_out_neurons,),
                                **self.lif_config)
        self.out_lif_right = LIF(shape=(self.num_out_neurons,),
                                 **self.lif_config)

        self.recv = ProcessOut(send_pipe=self.send_pipe,
                               frame_shape=self.frame_input.shape,
                               direction_shape=(self.num_out_neurons,))

        # Additionally implement Adapters in case loihi 2 is available
        if self.use_loihi2:
            self.in_adapter = PyToNxAdapter(
                shape=self.flat_shape, num_message_bits=8)
            self.out_adapter_left = NxToPyAdapter(shape=(self.num_out_neurons,))
            self.out_adapter_right = NxToPyAdapter(
                shape=(self.num_out_neurons,))

    def _make_connections(self) -> None:
        # Connect
        if self.use_loihi2:
            self.frame_input.s_out.connect(self.in_adapter.inp)
            self.in_adapter.out.connect(self.ff_inp.s_in)
            self.out_lif_left.s_out.connect(self.out_adapter_left.inp)
            self.out_adapter_left.out.connect(self.recv.direction_in_left)

            self.out_lif_right.s_out.connect(self.out_adapter_right.inp)
            self.out_adapter_right.out.connect(self.recv.direction_in_right)
        else:
            self.frame_input.s_out.connect(self.ff_inp.s_in)
            self.out_lif_right.s_out.connect(self.recv.direction_in_right)
            self.out_lif_left.s_out.connect(self.recv.direction_in_left)

        self.ff_inp.a_out.reshape(
            self.lif_inp.a_in.shape).connect(self.lif_inp.a_in)
        self.lif_inp.s_out.flatten().connect(self.ff_left.s_in)
        self.ff_left.a_out.reshape(
            self.lif_left.a_in.shape).connect(self.lif_left.a_in)
        self.lif_left.s_out.flatten().connect(self.rec_left.s_in)
        self.rec_left.a_out.reshape(
            self.lif_left.a_in.shape).connect(self.lif_left.a_in)
        self.lif_left.s_out.flatten().connect(self.sparse_out_left.s_in)
        self.sparse_out_left.a_out.connect(self.out_lif_left.a_in)
        self.lif_left.s_out.flatten().connect(self.sparse_out_left_inv.s_in)
        self.sparse_out_left_inv.a_out.connect(self.out_lif_right.a_in)
        self.lif_inp.s_out.flatten().connect(self.ff_right.s_in)
        self.ff_right.a_out.reshape(
            self.lif_right.a_in.shape).connect(self.lif_right.a_in)
        self.lif_right.s_out.flatten().connect(self.rec_right.s_in)
        self.rec_right.a_out.reshape(
            self.lif_right.a_in.shape).connect(self.lif_right.a_in)
        self.lif_right.s_out.flatten().connect(self.sparse_out_right.s_in)
        self.sparse_out_right.a_out.connect(self.out_lif_right.a_in)
        self.lif_right.s_out.flatten().connect(self.sparse_out_right_inv.s_in)
        self.sparse_out_right_inv.a_out.connect(self.out_lif_left.a_in)
        self.frame_input.s_out.reshape(
            self.frame_input.shape).connect(self.recv.frame_in)

    def start(self) -> None:
        self.runtime.start(RunSteps(num_steps=self.num_steps,
                                    blocking=self.blocking))

    def stop(self) -> None:
        self.runtime.wait()
        self.runtime.stop()
