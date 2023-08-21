# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort

import numpy as np
import typing as ty

from lava.proc.conv.utils import output_shape as compute_output_shape
from lava.lib.peripherals.realsense.convolution import conv


class RGBD2DepthSpaceTensor(AbstractProcess):
    """ Process converting Depth image into a Depth Space Tensor representation.

    At initialization, Depth space ([min_depth -> max_depth]) is divided into
    num_depth_bins bins.

    (1) An binned_depth_tensor ndarray (with shape (W, H, num_depth_bins))
        is initialized to zeros.
    (3) For every pixel (with index (x, y)) of the Depth image:
        If the Depth value of the pixel is >= min_depth and <= max_depth, set 1
        in binned_depth_tensor[x, y, idx], where idx corresponds to the bin
        where the Depth value of the pixel fits.
    (4) Apply an Average Filter
        (with kernel size given by down_sampling_factors)
        to binned_depth_tensor[:, :, idx] for idx in range(num_depth_bins).
    (5) Scale the result by max_bias.

    Parameters
    ----------
    shape_2d_in: tuple(int, int)
        2D shape of input image, corresponds to (W, H) where W is Width and
        H is height.
    down_sampling_factors: tuple(int, int)
        2D kernel size of the applied Average Filter.
    num_depth_bins: int
        Number of bins to divide the Depth space into.
    min_depth: float
        Depth value above which pixels are discarded.
    max_depth: float
        Depth value below which pixels are discarded.
    max_bias: int
        Scaling factor by which the result of the Average Filter is scaled.
    """
    def __init__(self,
                 shape_2d_in: ty.Tuple[int, int],
                 down_sampling_factors: ty.Tuple[int, int],
                 num_depth_bins: int,
                 min_depth: float,
                 max_depth: float,
                 max_bias: int) -> None:
        super().__init__(shape_2d_in=shape_2d_in,
                         down_sampling_factors=down_sampling_factors,
                         num_depth_bins=num_depth_bins,
                         min_depth=min_depth,
                         max_depth=max_depth,
                         max_bias=max_bias)

        input_shape = shape_2d_in + (1,)
        output_shape = compute_output_shape(input_shape=input_shape,
                                            out_channels=num_depth_bins,
                                            kernel_size=down_sampling_factors,
                                            stride=down_sampling_factors,
                                            padding=(0, 0),
                                            dilation=(1, 1))

        num_neurons = output_shape[0] * output_shape[1] * output_shape[2]
        down_sampled_flat_shape = (num_neurons,)

        self.a_in = InPort(shape=shape_2d_in)
        self.a_out = OutPort(shape=down_sampled_flat_shape)


@implements(proc=RGBD2DepthSpaceTensor, protocol=LoihiProtocol)
@requires(CPU)
class PyRGBD2DepthSpaceTensorProcModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params: ty.Dict) -> None:
        super().__init__(proc_params)

        self._shape_2d_in = proc_params["shape_2d_in"]
        self._down_sampling_factors = proc_params["down_sampling_factors"]
        self._num_depth_bins = proc_params["num_depth_bins"]
        self._min_depth = proc_params["min_depth"]
        self._max_depth = proc_params["max_depth"]
        self._max_bias = proc_params["max_bias"]

        bins_depth_ranges_start = np.linspace(self._min_depth, self._max_depth,
                                              num=self._num_depth_bins + 1)
        bins_depth_ranges = \
            list(zip(bins_depth_ranges_start[:-1], bins_depth_ranges_start[1:]))
        bins_depth_ranges[-1] = (
            bins_depth_ranges[-1][0], bins_depth_ranges[-1][1])
        self._bins_depth_ranges = bins_depth_ranges

        self._down_sampling_kernel = np.ones((self._num_depth_bins,
                                              self._down_sampling_factors[0],
                                              self._down_sampling_factors[1],
                                              self._num_depth_bins))

    def run_spk(self) -> None:
        depth_image = self.a_in.recv()
        binned_depth_tensor = self._extract_binned_depth_tensor(depth_image)
        down_sampled_binned_depth_tensor = \
            self._down_sample_binned_depth_tensor(binned_depth_tensor)
        print("depth_proc output", down_sampled_binned_depth_tensor.shape)
        self.a_out.send(down_sampled_binned_depth_tensor.flatten())

    def _extract_binned_depth_tensor(self,
                                     depth_image: np.ndarray) -> np.ndarray:
        """Extract Depth Space Tensor from Depth image.

        Parameters
        ----------
        depth_image: np.ndarray
            Depth image, of shape (W, H)

        Returns
        -------
        binned_depth_tensor: np.ndarray
            Depth Space Tensor, of shape (W, H, num_depth_bins)
        """
        binned_depth_tensor = np.zeros((self._shape_2d_in[0],
                                        self._shape_2d_in[1],
                                        self._num_depth_bins))

        for depth_bin_idx in range(self._num_depth_bins):
            left_condition = \
                depth_image >= self._bins_depth_ranges[depth_bin_idx][0]

            if depth_bin_idx != (self._num_depth_bins - 1):
                right_condition = \
                    depth_image < self._bins_depth_ranges[depth_bin_idx][1]
            else:
                right_condition = \
                    depth_image <= self._bins_depth_ranges[depth_bin_idx][1]

            idx_depth_in_bin_range = left_condition & right_condition

            binned_depth_tensor[idx_depth_in_bin_range, depth_bin_idx] = 1

        return binned_depth_tensor

    def _down_sample_binned_depth_tensor(self,
                                         binned_depth_tensor: np.ndarray) \
            -> np.ndarray:
        """Apply Average Filter and scale the result.

        Parameters
        ----------
        binned_depth_tensor: np.ndarray
            Depth Space Tensor, of shape (W, H, num_depth_bins)

        Returns
        -------
        down_sampled_binned_depth_tensor: np.ndarray
            Down-sampled Depth Space Tensor, of shape
            (W//down_sampling_factors[0],
             H//down_sampling_factors[1], num_depth_bins)
        """
        down_sampled_binned_depth_tensor = \
            conv(input_=binned_depth_tensor,
                 weight=self._down_sampling_kernel,
                 kernel_size=self._down_sampling_factors,
                 stride=self._down_sampling_factors,
                 padding=(0, 0),
                 dilation=(1, 1),
                 groups=1)

        down_sampling_factors_prod = \
            self._down_sampling_factors[0] * self._down_sampling_factors[1]

        down_sampled_binned_hsv_tensor = \
            np.rint(
                (down_sampled_binned_depth_tensor / down_sampling_factors_prod)
                * self._max_bias).astype(np.int32)

        return down_sampled_binned_hsv_tensor
