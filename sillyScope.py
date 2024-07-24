import libm2k
import numpy as np
import time

# Initialize context
ctx = libm2k.m2kOpen()
if ctx is None:
    print("Connection Error: No ADALM2000 device found")
    exit(1)

# Get the analog out channel
aout = ctx.getAnalogOut()

# Enable both channels (even if you only use one)
aout.enableChannel(0, True)
aout.enableChannel(1, True)

# Set the signal frequency
sampling_rate = 750000 # Set the sampling rate to 750 kHz

# Generate the signal
t = np.linspace(0, 6, int(sampling_rate * 6))  # 0 to 6 seconds
v = t  # v(t) = 0.8 * t

# Normalize the signal to fit within the -5V to +5V range of ADALM2000
#v = np.clip(v, 0, 5)

# Load the signal to the analog output buffer
aout.setCyclic(False)
aout.setSampleRate(0, sampling_rate)
aout.push(0, v.tolist())  # Channel 0 output

# Let it run for the duration of the signal
time.sleep(6)

# Disable channels after use
aout.enableChannel(0, False)
aout.enableChannel(1, False)

# Disconnect the device
libm2k.contextClose(ctx)
