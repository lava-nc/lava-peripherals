
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from multiprocessing import Event
import unittest
import numpy as np
import os
import copy

from lava.lib.peripherals.dvs.transform import Downsample, Compose, EventVolume, MergePolarities

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

        np.testing.assert_equal((events['x'] * factor).astype(int), downsampled_events['x'])
        np.testing.assert_equal((events['y'] * factor).astype(int), downsampled_events['y'])

    def test_per_channel(self):

        factor = {'x': 0.3, 'y':0.7}

        reader = RawReader(SEQUENCE_FILENAME_RAW)

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        downsample = Downsample(factor=factor)

        downsampled_events = copy.deepcopy(events)
        downsample(downsampled_events)

        np.testing.assert_equal((events['x'] * factor['x']).astype(int), downsampled_events['x'])
        np.testing.assert_equal((events['y'] * factor['y']).astype(int), downsampled_events['y'])

    def test_shape_transform(self):

        factor = 0.5

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height,
                                   width=width,
                                   polarities=2)

        downsample = Downsample(factor=factor)

        output_shape = downsample.determine_output_shape(input_shape=event_volume)

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

        self.assertTrue(np.all(merged_events['p'] == 0))

    def test_shape_transform(self):

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height,
                                   width=width,
                                   polarities=2)

        merge = MergePolarities()
        output_shape = merge.determine_output_shape(input_shape=event_volume)

        self.assertEqual(output_shape.polarities, 1)


class TestCompose(unittest.TestCase):

    def test_two_transforms(self):

        downsampling_factor = 0.5

        reader = RawReader(SEQUENCE_FILENAME_RAW)

        delta_t = 100000

        events = reader.load_delta_t(delta_t)
        transforms = Compose([MergePolarities(),
                              Downsample(factor=downsampling_factor)])

        transformed_events = copy.deepcopy(events)
        transforms(transformed_events)

        np.testing.assert_equal((events['x'] * downsampling_factor).astype(int), transformed_events['x'])
        np.testing.assert_equal((events['y'] * downsampling_factor).astype(int), transformed_events['y'])
        self.assertTrue(np.all(transformed_events['p'] == 0))

    def test_shape_transform(self):

        downsampling_factor = 0.5

        reader = RawReader(SEQUENCE_FILENAME_RAW)
        height, width = reader.get_size()
        event_volume = EventVolume(height=height,
                                   width=width,
                                   polarities=2)

        transforms = Compose([MergePolarities(),
                              Downsample(factor=downsampling_factor)])

        output_shape = transforms.determine_output_shape(event_volume)

        self.assertEqual(output_shape.width, int(width) * downsampling_factor)
        self.assertEqual(output_shape.height, int(height) * downsampling_factor)
        self.assertEqual(output_shape.polarities, 1)


