# Swipe Detection Demo

This readme assumes that you have installed lava, lava-loihi, lava-peripherals, and bokeh in the same virtual environment. 


Additionally, you should check if you downloaded the input recording which is stored using git lfs. Check if the file size is reasonable. Otherwise, run the following command.
```bash
lava-peripherals/demos/swipe_detection$ git lfs pull
```

### Running the demos
The demo will run in your browser via port-forwarding. Choose a random port_num between 10000 and 20000.
(This is to avoid that multiple users try to use the same port)

#### Connect to external vlab with port-forwarding
```bash
ssh <my-vm>.research.intel-research.net -L 127.0.0.1:<port_num>:127.0.0.1:<port_num>
```

#### Activate your virtual environment
Location of your virtual environment might differ.
```bash
source lava/lava_nx_env/bin/activate
```

#### Navigate to the swipe_detection demo:
```bash
cd lava-peripherals/demos/swipe_detection
```
#### Start the bokeh app
```bash
bokeh serve main_swipe_demo.py --port <port_num>
```

open your browser and type:
http://localhost:<port_num>/main_swipe_demo

As the network is pre-compiled, the demo will appear immediately, and you just need to click the "run" button to start the demo.
It is currently not possible to interrupt the demo while it is running. Please wait for the demo to terminate and click the "close" button for all processes to terminate gracefully. 
