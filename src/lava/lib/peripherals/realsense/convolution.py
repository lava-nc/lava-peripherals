# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_IS_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_IS_AVAILABLE = False
    from lava.proc.conv.utils import conv_scipy as conv

if TORCH_IS_AVAILABLE:
    def conv(input_: np.ndarray,
             weight: np.ndarray,
             kernel_size: Tuple[int, int],
             stride: Tuple[int, int],
             padding: Tuple[int, int],
             dilation: Tuple[int, int],
             groups: int) -> np.ndarray:
        """Convolution implementation

        Parameters
        ----------
        input_ : 3 dimensional np array
            convolution input.
        weight : 4 dimensional np array
            convolution kernel weight.
        kernel_size : 2 element tuple, list, or array
            convolution kernel size in XY/WH format.
        stride : 2 element tuple, list, or array
            convolution stride in XY/WH format.
        padding : 2 element tuple, list, or array
            convolution padding in XY/WH format.
        dilation : 2 element tuple, list, or array
            dilation of convolution kernel in XY/WH format.
        groups : int
            number of convolution groups.

        Returns
        -------
        3 dimensional np array
            convolution output
        """
        # with torch.no_grad():  # this seems to cause problems
        output = F.conv2d(
            torch.unsqueeze(  # torch expects a batch dimension NCHW
                torch.FloatTensor(input_.transpose([2, 1, 0])),
                dim=0,
            ),
            torch.FloatTensor(
                # torch actually does correlation
                # so flipping the spatial dimension of weight
                # copy() is needed because
                # torch cannot handle negative stride
                weight[:, ::-1, ::-1].transpose([0, 3, 2, 1]).copy()
            ),
            stride=list(stride[::-1]),
            padding=list(padding[::-1]),
            dilation=list(dilation[::-1]),
            groups=groups
        )[0].cpu().data.numpy().transpose([2, 1, 0])

        return output.astype(weight.dtype)
