# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import copy

from lava.lib.peripherals.dvs.transformation import (
    Downsample,
    Compose,
    EventVolume,
    MergePolarities,
    MirrorHorizontally,
    MirrorVertically,
)

from metavision_core.event_io import RawReader
from metavision_core.utils import get_sample


SEQUENCE_FILENAME_RAW = "sparklers.raw"
get_sample(SEQUENCE_FILENAME_RAW)


class TestDownsample(unittest.TestCase):
    def test_single_factor(self):
        factor = 0.5

        reader = RawReader(SEQUENCE_FILENAME_RAW)

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        downsample = Downsample(factor=factor)

        downsampled_events = copy.deepcopy(events)
        downsample(downsampled_events)

        np.testing.assert_equal(
            (events["x"] * factor).astype(int), downsampled_events["x"]
        )
        np.testing.assert_equal(
            (events["y"] * factor).astype(int), downsampled_events["y"]
        )

    def test_per_channel(self):
        factor = {"x": 0.3, "y": 0.7}

        reader = RawReader(SEQUENCE_FILENAME_RAW)

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        downsample = Downsample(factor=factor)

        downsampled_events = copy.deepcopy(events)
        downsample(downsampled_events)

        np.testing.assert_equal(
            (events["x"] * factor["x"]).astype(int), downsampled_events["x"]
        )
        np.testing.assert_equal(
            (events["y"] * factor["y"]).astype(int), downsampled_events["y"]
        )

    def test_shape_transform(self):
        factor = 0.5

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height, width=width, polarities=2)

        downsample = Downsample(factor=factor)

        output_shape = downsample.determine_output_shape(
            input_shape=event_volume
        )

        self.assertEqual(output_shape.width, int(width) * factor)
        self.assertEqual(output_shape.height, int(height) * factor)


class TestMergePolarities(unittest.TestCase):
    def test_merge(self):
        reader = RawReader(SEQUENCE_FILENAME_RAW)

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        merge = MergePolarities()

        merged_events = copy.deepcopy(events)
        merge(merged_events)

        self.assertTrue(np.all(merged_events["p"] == 0))

    def test_shape_transform(self):
        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height, width=width, polarities=2)

        merge = MergePolarities()
        output_shape = merge.determine_output_shape(input_shape=event_volume)

        self.assertEqual(output_shape.polarities, 1)


class TestMirrorHorizontally(unittest.TestCase):
    def test_mirror(self):
        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        transform = MirrorHorizontally(height=height)

        transformed_events = copy.deepcopy(events)
        transform(transformed_events)

        diff = events["y"] - height // 2
        desired_y = height // 2 - diff

        self.assertTrue(np.all(transformed_events["y"] == desired_y))

    def test_shape_transform(self):
        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height, width=width, polarities=2)

        transform = MirrorHorizontally(height=height)
        output_shape = transform.determine_output_shape(
            input_shape=event_volume
        )

        self.assertEqual(event_volume, output_shape)


class TestMirrorVertically(unittest.TestCase):
    def test_mirror(self):
        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        transform = MirrorVertically(width=width)

        transformed_events = copy.deepcopy(events)
        transform(transformed_events)

        diff = events["x"] - width // 2
        desired_x = width // 2 - diff

        self.assertTrue(np.all(transformed_events["x"] == desired_x))

    def test_shape_transform(self):
        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height, width=width, polarities=2)

        transform = MirrorVertically(width=width)
        output_shape = transform.determine_output_shape(
            input_shape=event_volume
        )

        self.assertEqual(event_volume, output_shape)


class TestCompose(unittest.TestCase):
    def test_two_transforms(self):
        downsampling_factor = 0.5

        reader = RawReader(SEQUENCE_FILENAME_RAW)

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        transforms = Compose(
            [MergePolarities(), Downsample(factor=downsampling_factor)]
        )

        transformed_events = copy.deepcopy(events)
        transforms(transformed_events)

        np.testing.assert_equal(
            (events["x"] * downsampling_factor).astype(int),
            transformed_events["x"],
        )
        np.testing.assert_equal(
            (events["y"] * downsampling_factor).astype(int),
            transformed_events["y"],
        )
        self.assertTrue(np.all(transformed_events["p"] == 0))

    def test_shape_transform(self):
        downsampling_factor = 0.5

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height, width=width, polarities=2)

        transforms = Compose(
            [MergePolarities(), Downsample(factor=downsampling_factor)]
        )

        output_shape = transforms.determine_output_shape(event_volume)

        self.assertEqual(output_shape.width, int(width) * downsampling_factor)
        self.assertEqual(output_shape.height, int(height) * downsampling_factor)
        self.assertEqual(output_shape.polarities, 1)
