# Lava Peripherals

Lava-peripherals is a library to the open-source framework [Lava](http://lava-nc.org) that adds support for peripheral devices such as cameras or robotic actuators.
Lava enables users to develop neuro-inspired applications and map them to neuromorphic hardware. It provides tools and abstractions to develop applications that fully exploit the principles of neural computation. 


# Content

- Dynamic Vision Cameras
  - Metavision
  - Inivation (coming soon)
- Intel RealSense cameras
- Robotic actuators (to be planned)
  

# Dependencies

Lava-peripherals currently requires Linux and does not support Windows or MacOS.

## Third-party dependencies

Lava-peripherals is flexible with the dependency on the libraries for the peripheral hardware and requires only those to be installed which are used. 

### PropheseeCamera
The `PropheseeCamera` Process, requires the [metavision-sdk](https://docs.prophesee.ai/stable/installation/index.html) v4.0.0 or newer to be installed. 

### RealSense
The `RealSense` Process, requires the [Intel® RealSense™ SDK](https://www.intelrealsense.com/sdk-2/) v2.0 and the Python wrapper [pyrealsense2](https://pypi.org/project/pyrealsense2/) to be 
installed. 

## Python dependencies

Lava-peripherals requires Python version 3.9 or newer; for installation either pip or poetry is required.

# Installation

## Linux

```bash
cd $HOME
curl -sSL https://install.python-poetry.org | python3 -
git clone https://github.com/lava-nc/lava-peripherals.git
cd lava-peripherals
poetry config virtualenvs.in-project true
poetry install

# in order to find metavision-sdk
sed -i "s/include-system-site-packages\ =\ false/include-system-site-packages\ =\ true/g" .venv/pyvenv.cfg

source .venv/bin/activate
pytest

## See FAQ for more info: https://github.com/lava-nc/lava/wiki/Frequently-Asked-Questions-(FAQ)#install
```

# More information

For more information visit http://lava-nc.org or the [Lava-nc on GitHub](https://github.com/lava-nc).
​

# Stay in touch

To receive regular updates on the latest developments and releases of the Lava
Software Framework
please [subscribe to our newsletter](http://eepurl.com/hJCyhb).
