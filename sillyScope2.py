import libm2k
import numpy as np
import time

# Initialize the device
adam = libm2k.m2kOpen()
if adam is None:
    print("Connection Error: No ADALM2000 device available/connected to your PC.")
    exit(1) #quits program

# Initialize analog output (signal generation)
aout = adam.getAnalogOut()
aout.enableChannel(0, True)

# Define the signal parameters
duration = 10  # Duration in seconds
total_sample_rate = 1000  # Total samples per second for output signal
num_output_samples = duration * total_sample_rate #6000

# Generate the time vector for output signal
t = np.linspace(0, 6, 6 * total_sample_rate) #6 second linear function

# Define the signal: v(t) = 0.8 * t
v_signal = 0.8 * t
v_signal = np.clip(v_signal, -5, 5)

# Set the signal to channel 0
aout.setCyclic(False)
aout.setSampleRate(0, total_sample_rate)
zeroes = [0] * 2000
combined_signal = zeroes + v_signal.tolist() + zeroes

aout.push(0, combined_signal) #aout contains the vector of signals to generate. we arranged it in a way so that it increases at a rate of 0.8 for every 0.001 second


#osilly scope time >:)
ain = adam.getAnalogIn()
# Initialize analog input (oscilloscope) for Channel 1
ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_1, True)  # Channel 1
ain.setSampleRate(total_sample_rate)  # sample rate
ain.setRange(libm2k.ANALOG_IN_CHANNEL_1, 0, 10)  # Channel 1 range

# Initialize analog input (oscilloscope) for Channel 2
ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_2, True)  # Channel 2
ain.setRange(libm2k.ANALOG_IN_CHANNEL_2, 0, 10)  # Channel 2 range


# ain1.setKernelBuffersCount(1) #allocate one kernel buffer to provide adequate memory for data acquisition
# ain2.setKernelBuffersCount(1)

# Record data for the duration of the signal
time.sleep(duration)

# Fetch the data
#data = ain.getSamples(6 * (total_sample_rate/10))  # Channel 1 data and Channel 2 data
data = ain.getSamples(6000)  # Channel 1 data and Channel 2 data

ain.stopAcquisition()

leSet = list(set(data[1]))

# Print or process the recorded data
print("Channel 1 Data:", leSet)  # Print all samples for Channel 1
print(f'\n Length: {len(leSet)}')
#print("Channel 2 Data:", data[1])  # Print all samples for Channel 2

# Close the connection
libm2k.contextClose(adam)
