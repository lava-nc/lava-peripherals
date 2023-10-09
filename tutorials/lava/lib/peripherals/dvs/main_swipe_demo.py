# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import sys
import threading
import functools
import multiprocessing

from lava.utils.serialization import load

try:
    from bokeh.plotting import figure, curdoc
    from bokeh.layouts import gridplot, Spacer
    from bokeh.models import LinearColorMapper, ColorBar, Title, Button
    from bokeh.models.ranges import DataRange1d
except ModuleNotFoundError:
    print("Module 'bokeh' is not installed. Please install module 'bokeh' in"
          " order to run the motion tracking demo.")
    exit()
from swipe_detection_network import SwipeDetector
from bokeh.models import Arrow, NormalHead
from bokeh.palettes import Muted3 as color
from lava.utils.system import Loihi2

# ==========================================================================
# Parameters
# ==========================================================================
recv_pipe, send_pipe = multiprocessing.Pipe()
num_steps = 400

# Checks whether terminate button has been clicked and allows to stop
# updating the bokeh doc
stop_button_pressed: bool = False
use_loihi2 = Loihi2.is_loihi2_available

_, executable = load("swipe_detector.pickle")
# ==========================================================================
# Set up network
# ==========================================================================
print("initializing network")
network = SwipeDetector(send_pipe,
                        num_steps,
                        use_loihi2,
                        executable=executable)
print("network initialized")
print(network.frame_input.shape)


# ==========================================================================
# Bokeh Helpers
# ==========================================================================
def callback_run() -> None:
    network.start()


def callback_stop() -> None:
    global stop_button_pressed
    stop_button_pressed = True
    network.stop()
    sys.exit()


def create_plot(plot_base_width,
                data_shape,
                title) -> (figure, figure.image):
    x_range = DataRange1d(start=0,
                          end=data_shape[0],
                          bounds=(0, data_shape[0]),
                          range_padding=50,
                          range_padding_units='percent')
    y_range = DataRange1d(start=0,
                          end=data_shape[1],
                          bounds=(0, data_shape[1]),
                          range_padding=50,
                          range_padding_units='percent')

    pw = plot_base_width
    ph = int(pw * (data_shape[1] / data_shape[0]))
    plot = figure(width=pw,
                  height=ph,
                  x_range=x_range,
                  y_range=y_range,
                  match_aspect=True,
                  tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                  toolbar_location=None)

    nh = NormalHead(fill_color=color[1],
                    fill_alpha=0.5,
                    size=10,
                    line_color=color[2])
    arrow = Arrow(end=nh, line_color=color[2], line_width=5,
                  x_start=data_shape[0] / 2, y_start=data_shape[1] / 2,
                  x_end=data_shape[0] / 2, y_end=data_shape[1] / 2)
    plot.add_layout(arrow, 'center')
    image = plot.image([], x=0, y=0, dw=data_shape[0], dh=data_shape[1],
                       palette="Viridis256", level="image")
    plot.add_layout(Title(text=title, align="center"), "above")
    x_grid = list(range(data_shape[0]))
    plot.xgrid[0].ticker = x_grid
    y_grid = list(range(data_shape[1]))
    plot.ygrid[0].ticker = y_grid
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    return plot, image, arrow


# ==========================================================================
# Instantiating Bokeh document
# ==========================================================================
bokeh_document = curdoc()

# create plots
dvs_frame_p, dvs_frame_im, arrow_bokeh = create_plot(
    400,
    (network.frame_input.shape[3], network.frame_input.shape[2]),
    "DVS file input (max pooling)")

# add a button widget and configure with the call back
button_run = Button(label="Run")
button_run.on_click(callback_run)

button_stop = Button(label="Close")
button_stop.on_click(callback_stop)
# finalize layout (with spacer as placeholder)
spacer = Spacer(height=40)
bokeh_document.add_root(
    gridplot([[button_run, None, button_stop],
              [None, spacer, None],
              [None, dvs_frame_p, None]],
             toolbar_options=dict(logo=None)))


# ==========================================================================
# Bokeh Update
# ==========================================================================
def update(dvs_frame, arrow) -> None:
    dvs_frame_im.data_source.data["image"] = [dvs_frame]
    arrow_bokeh.x_end = arrow[1][0]


# ==========================================================================
# Bokeh Main loop
# ==========================================================================
def main_loop() -> None:
    while not stop_button_pressed:
        if recv_pipe.poll():
            data_for_plot_dict = recv_pipe.recv()
            bokeh_document.add_next_tick_callback(
                functools.partial(update, **data_for_plot_dict))


thread = threading.Thread(target=main_loop)
thread.start()
