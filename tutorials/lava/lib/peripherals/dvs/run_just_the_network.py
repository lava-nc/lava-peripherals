# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from threading import Thread
import multiprocessing
from lava.utils.system import Loihi2
from swipe_detection_network_dummy import DummySwipeDetector

# ==========================================================================
# Parameters
# ==========================================================================
recv_pipe, send_pipe = multiprocessing.Pipe()
num_steps = 400

use_loihi2 = Loihi2.is_loihi2_available
print(use_loihi2)

# ==========================================================================
# Set up network
# ==========================================================================
print("initializing network")
network = DummySwipeDetector(send_pipe,
                        num_steps,
                        use_loihi2,
                        blocking=True)
print("network initialized")
print(network.frame_input.shape)

results = {}

def get_data(received_data: dict) -> None:
    for _ in range(num_steps):
        received_data[0] = recv_pipe.recv()

thread = Thread(target=get_data, args=[results])
thread.start()
network.start()
network.stop()