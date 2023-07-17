# Lava Peripherals

Lava is an open-source software framework for developing neuro-inspired applications and mapping them to neuromorphic hardware. Lava provides developers with the tools and abstractions to develop applications that fully exploit the principles of neural computation. 
Lava-peripherals is a library that connects peripheral hardware such as cameras or robotic actuators to [Lava](http://lava-nc.org).​

# Content

- Dynamic Vision Cameras
  - Metavision
  - Inivation (coming soon)
- Intel RealSense cameras (coming soon)
- Robotic actuators (coming soon)
  

# Dependencies

Lava-peripherals currently requires Linux and does not support Windows or MacOS.

## Third-party dependencies

Lava-peripherals is flexible with the dependency on the libraries for the peripheral hardware and requires only those to be installed which are used. 

### PropheseeCamera
In order to run the `PropheseeCamera` Process, the [metavision-sdk](https://docs.prophesee.ai/stable/installation/index.html) is required to be installed. 

## Python dependencies

Lava-peripherals requires Python version 3.9 or newer and for installation either pip or poetry is requrired.

# Installation

## Linux

```bash
cd $HOME
curl -sSL https://install.python-poetry.org | python3 -
git clone https://github.com/lava-nc/lava-peripherals.git
cd lava-peripherals
poetry config virtualenvs.in-project true
poetry config virtualenvs.options.system-site-packages true # in order to find metavision-sdk
poetry install
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
