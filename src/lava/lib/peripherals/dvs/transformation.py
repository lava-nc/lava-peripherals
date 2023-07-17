# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import copy
import typing as ty


@dataclass
class EventVolume:
    height: int
    width: int
    polarities: int
    t_start: int = None
    t_stop: int = None


class Transformation:
    """Base class for transformations."""

    @abstractmethod
    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Transform data.

        Parameters
        ----------
        events: np.ndarray
            DVS events

        Returns
        -------
        events: np.ndarray
            Transformed DVS events

        """
        pass

    @abstractmethod
    def determine_output_shape(self, input_shape: EventVolume) -> EventVolume:
        """Transform the shape of the events.

        Parameters
        ----------
        input_shape: EventVolume
            Shape of the incoming events.

        Returns
        -------
        output_shape: EventVolume
            Shape of the outcoming events.
        """
        pass


class Compose:
    def __init__(self, transformations: ty.Iterable[Transformation]):
        """Allows to combine transformations similar to torchvision.Compose.

        Parameters
        ----------
        transformations: Iterable
            Iterable of transformations
        """
        self.transformations = transformations

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Apply all transformations:

        Parameters
        ----------
        events: np.ndarray
            DVS events

        Returns
        -------
        events: np.ndarray
            Transformed DVS events
        """
        for t in self.transformations:
            events = t(events)
        return events

    def determine_output_shape(self, input_shape: EventVolume):
        """Apply shape transformation of all transfomations in this compose.

        Parameters
        ----------
        input_shape: EventVolume
           Shape of the incoming events.

        Returns
        -------
        output_shape: EventVolume
            Shape of the outcoming events.
        """
        for t in self.transformations:
            input_shape = t.determine_output_shape(input_shape)
        return input_shape


class Downsample(Transformation):
    def __init__(self, factor: ty.Union[float, ty.Dict[str, float]]):
        """Downsamples x and y coordinates by given factor.

        Parameters
        ----------
        factor: float, List[float]
            The factor by which x and y coordinates are downsampled. Can be
            float if the factor shall be the same for x and y, dict defining
            'x' and 'y'.
        input_shape: EventVolume
            Shape of the incoming events.
        output_shape: EventVolume
            If not provided, the transformation calculates the output shape.
            automatically.
        """
        super().__init__()
        if isinstance(factor, float):
            self.factor_x = self.factor_y = factor
        elif isinstance(factor, dict):
            self.factor_x = factor["x"]
            self.factor_y = factor["y"]
        else:
            raise NotImplementedError(
                "factor must be either of type float or dict."
            )

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Transform data by multiplying height and width with given factor.

        Parameters
        ----------
        events: np.ndarray
            DVS events
        Returns
        -------
        events: np.ndarray
            Transformed DVS events
        """
        events["x"] = (events["x"] * self.factor_x).astype(np.int32)
        events["y"] = (events["y"] * self.factor_y).astype(np.int32)
        return events

    def determine_output_shape(self, input_shape: EventVolume) -> EventVolume:
        """Determine output shape.

         Parameters
         ----------
         input_shape: EventVolume
             Shape of the incoming events.

        Returns
         -------
         output_shape: EventVolume
             Shape of the outcoming events.
        """
        output_shape = copy.deepcopy(input_shape)
        output_shape.width = int(output_shape.width * self.factor_x)
        output_shape.height = int(output_shape.height * self.factor_y)
        return output_shape


class MergePolarities(Transformation):
    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Put all events in one polarity.

        Parameters
        ----------
        events: np.ndarray
            DVS events
        Returns
        -------
        events: np.ndarray
            Transformed DVS events
        """
        events["p"] = (events["p"] * 0).astype(np.int32)
        return events

    def determine_output_shape(self, input_shape: EventVolume) -> EventVolume:
        """Determine output shape.

        Parameters
        ----------
        input_shape: EventVolume
            Shape of the incoming events.

        Returns
        -------
        output_shape: EventVolume
            Shape of the outcoming events.
        """
        output_shape = copy.deepcopy(input_shape)
        output_shape.polarities = 1
        return output_shape


class MirrorHorizontally(Transformation):
    def __init__(self, height: int):
        """Mirror events on a horizontal axis.

        Parameters
        ----------
        height: int
            Height of the original events.
        """
        super().__init__()
        self.height = height

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Mirror events on a horizontal axis.

        Parameters
        ----------
        events: np.ndarray
            DVS events
        Returns
        -------
        events: np.ndarray
            Transformed DVS events
        """
        events["y"] = self.height - events["y"]
        return events

    def determine_output_shape(self, input_shape: EventVolume) -> EventVolume:
        """Determine output shape.

        Parameters
        ----------
        input_shape: EventVolume
            Shape of the incoming events.

        Returns
        -------
        output_shape: EventVolume
            Shape of the outcoming events.
        """
        return input_shape


class MirrorVertically(Transformation):
    def __init__(self, width: int):
        """Mirror events on a vertical axis.

        Parameters
        ----------
        width: int
           Width of the original events.
        """
        super().__init__()
        self.width = width

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """Mirror events on a vertical axis.

        Parameters
        ----------
        events: np.ndarray
            DVS events
        Returns
        -------
        events: np.ndarray
            Transformed DVS events
        """
        events["x"] = self.width - events["x"]
        return events

    def determine_output_shape(self, input_shape: EventVolume) -> EventVolume:
        """Determine output shape.

        Parameters
        ----------
        input_shape: EventVolume
            Shape of the incoming events.

        Returns
        -------
        output_shape: EventVolume
            Shape of the outcoming events.
        """
        return input_shape
