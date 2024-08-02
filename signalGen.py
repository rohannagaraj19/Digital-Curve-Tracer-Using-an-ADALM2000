import libm2k
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.signal as sig
#this is code to play around with different time signals
#this will input a signal of your choosing and return an graph of the oscilloscope readings
#graphs a voltage and time graph


# Initialize the device
uri = libm2k.getAllContexts()
if uri is None:
    print("Connection Error: No ADALM2000 device available/connected to your PC.")
    exit(1) #quits program
else:
    print("Successfully connected ADALM2000...")

adam = libm2k.m2kOpen(uri[0])
adam.reset() #resets the kernel buffer (MUST HAVE FOR CALIBRATION)
adam.calibrate() #sets the kernel buffer for the oscilloscope
adam.calibrateDAC() #digital to analog converter calibration (sets the kernel buffer)
print("Calibration successful!")

# Define the signal parameters
duration = 6  # Duration in seconds
scope_sample_rate = 1000  #sample rate of the oscilloscope
signal_sample_rate = 75000 # Total samples per second for output signal (must surpass Nyquist Sample rate to prevent aliasing)
num_output_samples = duration * scope_sample_rate 

aout = adam.getAnalogOut()
aout.enableChannel(0, True)
t = np.linspace(0, duration, signal_sample_rate * duration)

v_signal = 4*np.sin(2*np.pi *t)  #input signal equation


v_signal = np.clip(v_signal, 0, 5) #the adalm can only push a voltage of 5V... max current with amplifier is around 500 mA
aout.setCyclic(False)
aout.setSampleRate(0, signal_sample_rate) #param: channel (0 means channel 1), sampling frequency of input signal (typically 1000 Hz but needs to be greater than the Nyquist rate)
aout.push(0, v_signal.tolist())

# Oscilloscope setup
ain = adam.getAnalogIn()
ain.setSampleRate(scope_sample_rate)  # sample rate
# Initialize analog input (oscilloscope) for Channel 1
ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_1, True)  # Channel 1
ain.setRange(libm2k.ANALOG_IN_CHANNEL_1, 0, 5)  # Channel 1 range

# Initialize analog input (oscilloscope) for Channel 2
ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_2, True)  # Channel 2
ain.setRange(libm2k.ANALOG_IN_CHANNEL_2, 0, 5)  # Channel 2 range

voltageDiode, voltageInput = np.array(ain.getSamples(num_output_samples))
time_x = np.linspace(0, duration, len(voltageDiode))

fig, ax = plt.subplots()
ax.plot(time_x, voltageDiode, label=f'Diode Voltage (V)', color = "red")
ax.plot(time_x, voltageInput, label=f'Diode Input Voltage (V)', color = "purple")
ax.grid()
ax.set_xlabel('Time(s)')
ax.set_ylim([0, 7])
ax.set_ylabel("Voltage")
ax.legend()
plt.show()


ain.stopAcquisition()
libm2k.contextClose(adam)
