import typing as ty
import numpy as np
from multiprocessing import Pipe
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.runtime.runtime import Runtime
from lava.lib.peripherals.dvs.prophesee import PropheseeCamera
from lava.lib.peripherals.dvs.transform import Compose, MergePolarities, \
    Downsample
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.sparse.process import Sparse
from lava.proc.dense.models import PyDenseModelBitAcc
from lava.proc.lif.models import PyLifModelBitAcc
from lava.proc.sparse.models import PySparseModelBitAcc
from lava.proc.embedded_io.spike import NxToPyAdapter, PyToNxAdapter

from outprocess.process_out import ProcessOut
from scipy.sparse import csr_matrix

prophesee_input_default_config = {
    "filename": "hand_swipe.dat",
    "transformations": Compose(
        [
            Downsample(factor=0.05),
            MergePolarities(),
        ]
    ),
    "sensor_shape": (480, 640),
    "num_output_time_bins": 1,
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

num_out_neurons_default = 50
blocking_default_config = True


class SwipeDetector():
    def __init__(self,
                 send_pipe: type(Pipe),
                 num_steps: int,
                 use_loihi2: bool,
                 prophesee_input_config: ty.Optional[dict] = None,
                 lif_config: ty.Optional[dict] = None,
                 weights_config: ty.Optional[dict] = None,
                 num_out_neurons: ty.Optional[dict] = None,
                 blocking: ty.Optional[bool] = False,
                 ):
        self.prophesee_input_config = prophesee_input_config or \
                                      prophesee_input_default_config
        self.lif_config = lif_config or lif_default_config
        self.weights_config = weights_config or weights_default_config
        self.num_out_neurons = num_out_neurons or num_out_neurons_default
        self.blocking = blocking or blocking_default_config
        self.send_pipe = send_pipe
        self.num_steps = num_steps
        self.use_loihi2 = use_loihi2

        print("hi")
        self.create_processes()
        self.make_connections()

        print("network created")
        # Run
        if self.use_loihi2:
            print("running on chip")
            run_cfg = Loihi2HwCfg(
                exception_proc_model_map={})
        else:
            run_cfg = Loihi2SimCfg(
                exception_proc_model_map={Dense: PyDenseModelBitAcc, Sparse: PySparseModelBitAcc, LIF: PyLifModelBitAcc})

        # Compilation
        compiler = Compiler()
        executable = compiler.compile(self.frame_input, run_cfg=run_cfg)

        # Initialize runtime
        mp = ActorType.MultiProcessing
        self.runtime = Runtime(exe=executable,
                               message_infrastructure_type=mp)
        self.runtime.initialize()
        print("done here")

    def create_processes(self):
        # Create Processes and Weights
        self.frame_input = PropheseeCamera(**self.prophesee_input_config)
        self.num_neurons = np.prod(self.frame_input.s_out.shape)
        _, _, self.scaled_height, self.scaled_width = self.frame_input.shape
        self.flat_shape = (self.num_neurons,)
        print(self.num_neurons)

        # create weights
        # FF weights
        self.ff_weights = np.eye(self.num_neurons) * self.weights_config["w_ff"]

        # Left weights
        self.rec_weights_left = np.zeros((self.num_neurons, self.num_neurons)).reshape((self.scaled_height, self.scaled_width, self.scaled_height, self.scaled_width))

        for i in range(self.scaled_height):
            for j in range(self.scaled_width):
                self.rec_weights_left[i, j - self.weights_config["kernel_size"]:j, i, j] = self.weights_config["w_rec"]

        self.rec_weights_left = csr_matrix(self.rec_weights_left.reshape((self.num_neurons, self.num_neurons)))
        print(self.rec_weights_left.shape)

        # right weights
        self.rec_weights_right = np.zeros((self.num_neurons, self.num_neurons)).reshape((self.scaled_height, self.scaled_width, self.scaled_height, self.scaled_width))
        for i in range(self.scaled_height):
            for j in range(self.scaled_width):
                self.rec_weights_right[i, j + 1:j + self.weights_config["kernel_size"] + 1, i, j] = \
                    self.weights_config["w_rec"]

        self.rec_weights_right = csr_matrix(self.rec_weights_right.reshape((self.num_neurons, self.num_neurons)))

        # Weights out
        self.w_out_left = np.zeros((2 * self.num_out_neurons, self.num_neurons))
        self.w_out_right = np.zeros((2 * self.num_out_neurons, self.num_neurons))

        tmp = np.kron(np.eye(self.num_out_neurons, dtype=np.int32), np.array([self.weights_config["w_o"]]
                                                                             * (
                                                                                         self.num_neurons // self.num_out_neurons)))

        self.w_out_left[:self.num_out_neurons, :tmp.shape[1]] = tmp
        self.w_out_right[self.num_out_neurons:2 * self.num_out_neurons, :tmp.shape[1]] = tmp

        self.w_out_left[:, 0] = 1  # bug fix
        self.w_out_right[:, 0] = 1  # bug fix

        print("Create inp processes")
        self.ff_inp = Sparse(weights=self.ff_weights, num_message_bits=8)
        self.lif_inp = LIF(shape=self.frame_input.shape, **self.lif_config)

        print("Create left processes")
        self.ff_left = Sparse(weights=self.ff_weights)
        self.rec_left = Sparse(weights=self.rec_weights_left)
        self.lif_left = LIF(shape=self.frame_input.shape, **self.lif_config)

        print("Create right processes")
        self.ff_right = Sparse(weights=self.ff_weights)
        self.rec_right = Sparse(weights=self.rec_weights_right)
        self.lif_right = LIF(shape=self.frame_input.shape, **self.lif_config)

        print("Create out processes")

        self.sparse_out_left = Sparse(weights=self.w_out_left)
        self.sparse_out_right = Sparse(weights=self.w_out_right)

        self.sparse_out_left_inv = Sparse(weights=-self.w_out_right)
        self.sparse_out_right_inv = Sparse(weights=-self.w_out_left)

        self.out_lif = LIF(shape=(2 * self.num_out_neurons,), **self.lif_config)

        self.recv = ProcessOut(send_pipe=self.send_pipe,
            shape=self.frame_input.shape,
                                direction_shape=(2 * self.num_out_neurons,))

        ## Additionaly implement Adapters in case loihi 2 is available
        if self.use_loihi2:
            self.in_adapter = PyToNxAdapter(shape=self.flat_shape)
            self.out_adapter = NxToPyAdapter(shape=(2 * self.num_out_neurons,))

    def make_connections(self):
        print("Connect")
        # Connect

        if self.use_loihi2:
            self.frame_input.s_out.connect(self.in_adapter.inp)
            self.in_adapter.out.connect(self.ff_inp.s_in)
            self.out_lif.s_out.connect(self.out_adapter.inp)
            self.out_adapter.out.connect(self.recv.direction_in)
        else:
            self.frame_input.s_out.flatten().connect(self.ff_inp.s_in)
            self.out_lif.s_out.connect(self.recv.direction_in)

        self.ff_inp.a_out.reshape(self.lif_inp.a_in.shape).connect(self.lif_inp.a_in)
        self.lif_inp.s_out.flatten().connect(self.ff_left.s_in)
        self.ff_left.a_out.reshape(self.lif_left.a_in.shape).connect(self.lif_left.a_in)
        self.lif_left.s_out.flatten().connect(self.rec_left.s_in)
        self.rec_left.a_out.reshape(self.lif_left.a_in.shape).connect(self.lif_left.a_in)
        self.lif_left.s_out.flatten().connect(self.sparse_out_left.s_in)
        self.sparse_out_left.a_out.connect(self.out_lif.a_in)
        self.lif_left.s_out.flatten().connect(self.sparse_out_left_inv.s_in)
        self.sparse_out_left_inv.a_out.connect(self.out_lif.a_in)

        self.lif_inp.s_out.flatten().connect(self.ff_right.s_in)
        self.ff_right.a_out.reshape(self.lif_right.a_in.shape).connect(self.lif_right.a_in)
        self.lif_right.s_out.flatten().connect(self.rec_right.s_in)
        self.rec_right.a_out.reshape(self.lif_right.a_in.shape).connect(self.lif_right.a_in)
        self.lif_right.s_out.flatten().connect(self.sparse_out_right.s_in)
        self.sparse_out_right.a_out.connect(self.out_lif.a_in)
        self.lif_right.s_out.flatten().connect(self.sparse_out_right_inv.s_in)
        self.sparse_out_right_inv.a_out.connect(self.out_lif.a_in)
        self.frame_input.s_out.reshape(self.frame_input.shape).connect(self.recv.frame_in)




    def start(self) -> None:
        self.runtime.start(RunSteps(num_steps=self.num_steps, blocking=self.blocking))

    def stop(self) -> None:
        self.runtime.wait()
        self.runtime.stop()
