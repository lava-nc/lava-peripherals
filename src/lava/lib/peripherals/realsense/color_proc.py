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

import cv2

from lava.proc.conv.utils import output_shape as compute_output_shape
from lava.lib.peripherals.realsense.convolution import conv


class BGR2HueSpaceTensor(AbstractProcess):
    """ Process converting BGR image into a Hue Space Tensor representation.

    At initialization, Hue space ([0 -> 180]) is divided into num_color_bins
    bins.

    (1) BGR image (with shape (W, H, 3)) is converted into
        HSV image (with shape (W, H, 3)).
    (2) An binned_hsv_tensor ndarray (with shape (W, H, num_color_bins))
        is initialized to zeros.
    (3) For every pixel (with index (x, y)) of the HSV image:
        If the Saturation value of the pixel is >= saturation_threshold, set 1
        in binned_hsv_tensor[x, y, idx], where idx corresponds to the bin where
        the Hue value of the pixel fits.
    (4) Apply an Average Filter (with kernel size given by down_sampling_factors)
        to binned_hsv_tensor[:, :, idx] for idx in range(num_color_bins).
    (5) Scale the result by max_bias.

    Parameters
    ----------
    shape_2d_in: tuple(int, int)
        2D shape of input image, corresponds to (W, H) where W is Width and
        H is height.
    down_sampling_factors: tuple(int, int)
        2D kernel size of the applied Average Filter.
    num_color_bins: int
        Number of bins to divide the Hue space into.
    saturation_threshold: int
        Saturation value below which pixels are discarded.
    max_bias: int
        Scaling factor by which the result of the Average Filter is scaled.
    """
    def __init__(self,
                 shape_2d_in: ty.Tuple[int, int],
                 down_sampling_factors: ty.Tuple[int, int],
                 num_color_bins: int,
                 saturation_threshold: int,
                 max_bias: int) -> None:
        super().__init__(shape_2d_in=shape_2d_in,
                         down_sampling_factors=down_sampling_factors,
                         num_color_bins=num_color_bins,
                         saturation_threshold=saturation_threshold,
                         max_bias=max_bias)

        input_shape = shape_2d_in + (3,)
        output_shape = compute_output_shape(input_shape=input_shape,
                                            out_channels=num_color_bins,
                                            kernel_size=down_sampling_factors,
                                            stride=down_sampling_factors,
                                            padding=(0, 0),
                                            dilation=(1, 1))
        num_neurons = output_shape[0] * output_shape[1] * output_shape[2]
        down_sampled_flat_shape = (num_neurons,)

        self.a_in = InPort(shape=input_shape)
        self.a_out = OutPort(shape=down_sampled_flat_shape)


@implements(proc=BGR2HueSpaceTensor, protocol=LoihiProtocol)
@requires(CPU)
class PyBGR2HueSpaceTensorProcModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params: ty.Dict) -> None:
        super().__init__(proc_params)

        self._shape_2d_in = proc_params["shape_2d_in"]
        self._down_sampling_factors = proc_params["down_sampling_factors"]
        self._num_color_bins = proc_params["num_color_bins"]
        self._saturation_threshold = proc_params["saturation_threshold"]
        self._max_bias = proc_params["max_bias"]

        bins_hue_ranges_start = np.linspace(0, 180,
                                            num=self._num_color_bins + 1)
        bins_hue_ranges = \
            list(zip(bins_hue_ranges_start[:-1], bins_hue_ranges_start[1:]))
        bins_hue_ranges[-1] = (
            bins_hue_ranges[-1][0], bins_hue_ranges[-1][1] + 1)
        self._bins_hue_ranges = bins_hue_ranges

        self._down_sampling_kernel = np.ones((self._num_color_bins,
                                              self._down_sampling_factors[0],
                                              self._down_sampling_factors[1],
                                              self._num_color_bins))

    def run_spk(self) -> None:
        bgr_image = self.a_in.recv().astype(np.uint8)
        binned_hsv_tensor = self._extract_binned_hsv_tensor(bgr_image)
        down_sampled_binned_hsv_tensor = \
            self._down_sample_binned_hsv_tensor(binned_hsv_tensor)
        # Simple printing of HSV Tensor Output
        # print("color_proc output : ", down_sampled_binned_hsv_tensor)

        self.a_out.send(down_sampled_binned_hsv_tensor.flatten())

    def _extract_binned_hsv_tensor(self, bgr_image: np.ndarray) -> np.ndarray:
        """Extract Hue Space Tensor from BGR image.

        Parameters
        ----------
        bgr_image: np.ndarray
            BGR image, of shape (W, H, 3)

        Returns
        -------
        binned_hsv_tensor: np.ndarray
            Hue Space Tensor, of shape (W, H, num_color_bins)
        """
        hsv_image = cv2.cvtColor(bgr_image.astype(np.uint8), cv2.COLOR_BGR2HSV)

        binned_hsv_tensor = np.zeros((self._shape_2d_in[0],
                                      self._shape_2d_in[1],
                                      self._num_color_bins))

        idx_saturation_above_threshold = \
            hsv_image[:, :, 1] >= self._saturation_threshold

        for color_bin_idx in range(self._num_color_bins):
            idx_hue_in_bin_range = (hsv_image[:, :, 0] >= \
                                    self._bins_hue_ranges[color_bin_idx][
                                        0]) & (hsv_image[:, :, 0] < \
                                               self._bins_hue_ranges[
                                                   color_bin_idx][1])

            idx_condition = idx_saturation_above_threshold & \
                            idx_hue_in_bin_range

            binned_hsv_tensor[idx_condition, color_bin_idx] = 1

        return binned_hsv_tensor

    def _down_sample_binned_hsv_tensor(self,
                                       binned_hsv_tensor: np.ndarray) \
            -> np.ndarray:
        """Apply Average Filter and scale the result.

        Parameters
        ----------
        binned_hsv_tensor: np.ndarray
            Hue Space Tensor, of shape (W, H, num_color_bins)

        Returns
        -------
        down_sampled_binned_hsv_tensor: np.ndarray
            Down-sampled Hue Space Tensor, of shape
            (W//down_sampling_factors[0], H//down_sampling_factors[1], num_color_bins)
        """
        down_sampled_binned_hsv_tensor = \
            conv(input_=binned_hsv_tensor,
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
                (down_sampled_binned_hsv_tensor / down_sampling_factors_prod)
                * self._max_bias).astype(np.int32)

        return down_sampled_binned_hsv_tensor
