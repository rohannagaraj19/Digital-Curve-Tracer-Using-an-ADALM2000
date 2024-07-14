import libm2k
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize the device
adam = libm2k.m2kOpen()
if adam is None:
    print("Connection Error: No ADALM2000 device available/connected to your PC.")
    exit(1) #quits program
else:
    print("Successfully connected ADALM2000...")

# Define the signal parameters
duration = 6  # Duration in seconds
scope_sample_rate = 1000  #sample rate of the oscilloscope
signal_sample_rate = 1000 # Total samples per second for output signal (must surpass Nyquist Sample rate to prevent aliasing)
num_output_samples = duration * 1000 

aout = adam.getAnalogOut()
aout.enableChannel(0, True)
t = np.linspace(0, duration, signal_sample_rate*duration)
v_signal = 0.8*t #input signal equation
v_signal = np.clip(v_signal, -5, 5)
aout.setCyclic(False)
aout.setSampleRate(0, signal_sample_rate) #param: channel (0 means channel 1), sampling frequency of input signal (typically 1000 Hz but needs to be greater than the Nyquist rate)
aout.push(0, v_signal.tolist())


#osilly scope time >:)
ain = adam.getAnalogIn()
ain.setSampleRate(scope_sample_rate)  # sample rate
# Initialize analog input (oscilloscope) for Channel 1
ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_1, True)  # Channel 1
ain.setRange(libm2k.ANALOG_IN_CHANNEL_1, 0, 5)  # Channel 1 range

# Initialize analog input (oscilloscope) for Channel 2
ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_2, True)  # Channel 2
ain.setRange(libm2k.ANALOG_IN_CHANNEL_2, 0, 5)  # Channel 2 range

#graph configurations
#first delete duplicate input voltages
diodeV, inputV = np.array(ain.getSamples(num_output_samples)) #y_data[0] has all the channel 1 data #y_data[1] has all the channel 2 data
inputV, indices1 = np.unique(inputV, return_index= True)
diodeV = diodeV[indices1]
#time characteristics
time_x = np.linspace(0, duration, len(diodeV))

fig, ax = plt.subplots()
#plt.scatter(time_x, y_data[0], marker = '.', s=50, edgecolor = "black", c= "black")
plt.plot(time_x, diodeV, color = 'indigo', label = 'Diode Voltage (V)')
plt.plot(time_x, inputV, color = 'lightcoral', label = 'Input Voltage (V)')
ax.grid()
ax.set_xlabel('Time(s)')
ax.set_ylim([0,5])
ax.set_ylabel("Voltage")
plt.legend()
plt.savefig('vtGraph.png')
#plt.show()

ain.stopAcquisition()

#data analysis time :(
resistance = 100  #what resistance is in series with the diode
current_data = list(((inputV - diodeV)/resistance for inputV, diodeV in zip(inputV, diodeV)))
current_data = np.array(current_data)
#delete duplicate values of current now
diodeV, indices = np.unique(diodeV, return_index=True)
current_data = current_data[indices]

fig2, bx = plt.subplots()
plt.scatter(diodeV, current_data, marker = '.', s=20, edgecolor = 'black', c= 'black', label = 'IV Curve')
bx.grid()
bx.set_xlabel("Voltage Across Diode (V)")
bx.set_ylim([0,0.05])
bx.set_ylabel("Input Current (A)")
plt.legend()
plt.savefig('ivGraph.png')
plt.show()
