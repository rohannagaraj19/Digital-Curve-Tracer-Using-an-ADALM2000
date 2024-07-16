import libm2k
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def CurveTrace(name, color):
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
    #inputV, indices1 = np.unique(inputV, return_index= True)
    #diodeV = diodeV[indices1]
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
    #diodeV, indices = np.unique(diodeV, return_index=True)
    #current_data = current_data[indices]
    diodeV = np.clip(diodeV, a_min = 0, a_max = 1000000)
    #find equation that fits
    def Exp(V, I0, Vt, V_off):
        return I0 * np.exp(V*Vt) + V_off
    param, cov = curve_fit(f= Exp, xdata = diodeV, ydata = current_data, p0 = (0,1,0.1))
    I0, Vt, V_off = param
    current_fit = np.linspace(0, 0.05, 100)
    voltage_fit = (np.log((current_fit-V_off)/I0)/Vt)

    fig2, bx = plt.subplots()
    plt.scatter(diodeV, current_data, marker = '.', s=20, edgecolor = 'black', c= 'black')
    plt.plot(voltage_fit, current_fit, color, label = name)
    bx.grid()
    bx.set_xlabel("Voltage Across Diode (V)")
    bx.set_ylim([0,0.05])
    bx.set_ylabel("Input Current (A)")
    plt.legend()
    plt.savefig(f'diode_graphs/{name}.png')
    plt.show()
    libm2k.contextClose(adam)

CurveTrace("R0xB", "blue")
halt = str(input("Please configure next diode then type 'y' then 'enter'"))
if(halt == 'y'):
    CurveTrace("R0xA", "green")
else:
    exit(1)