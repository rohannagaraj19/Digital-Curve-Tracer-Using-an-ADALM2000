import libm2k
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpld3

def CurveTrace(name):
    fig, ax = plt.subplots()
    fig2, bx = plt.subplots()
    colors = ['red', 'blue', 'green']
    # Fit data to exponential model
    def Exp(V, I0, Vt, V_off):
        return I0 * np.exp(V * Vt) + V_off
    diodeV = []
    for i in range(3):
        diodeV.append([])
    
    current_data = []
    for i in range(3):
        current_data.append([])
    
    inputV = []
    for i in range(3):
        inputV.append([])

    for i in range(1, 4, 1): #iterate three times
        halt = str(input(f"Please configure diode {i} for tracing. Press 'y' and 'enter' when ready..."))
        if halt != 'y': 
            exit(1) #quits program

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
        t = np.linspace(0, duration, signal_sample_rate * duration)
        v_signal = 0.8 * t  #input signal equation
        v_signal = np.clip(v_signal, 0, 5)
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

        # Collect and process data
        diodeV[i-1], inputV[i-1] = np.array(ain.getSamples(num_output_samples)) #diodeV has all the channel 1 data and inputV has all the channel 2 data
        print(f"diodeV length {len(diodeV[i-1])}")
        print(f"inputV length {len(inputV[i-1])}")
        # First delete duplicate input voltages
        inputV[i-1], indices1 = np.unique(inputV[i-1], return_index=True)
        diodeV[i-1] = diodeV[i-1][indices1]
        # Time characteristics
        time_x = np.linspace(0, duration, len(diodeV[i-1]))

        # Plot voltage-time graph
        ax.plot(time_x, diodeV[i-1], label=f'Diode {i} Voltage (V)')
        ax.plot(time_x, inputV[i-1], label=f'Diode {i} Input Voltage (V)')
        
        # Calculate current and voltage for IV curve
        resistance = 100  #what resistance is in series with the diode
        current_data[i-1] = list((inputV - diodeV) / resistance for inputV, diodeV in zip(inputV[i-1], diodeV[i-1]))
        current_data[i-1] = np.array(current_data[i-1]) #convert to an numpy array
        # Delete duplicate values of current now
        diodeV[i-1], indices = np.unique(diodeV[i-1], return_index=True)
        current_data[i-1] = current_data[i-1][indices]
        diodeV[i-1] = np.clip(diodeV[i-1], a_min=0, a_max=10e6)



        #curve fitting
        param, _ = curve_fit(f=Exp, xdata=diodeV[i-1], ydata=current_data[i-1], p0=(0, 1, 0.1))
        I0, Vt, V_off = param
        current_fit = np.linspace(0, 0.05, 100)
        voltage_fit = (np.log((current_fit - V_off) / I0) / Vt)

        # Plot IV curve
        bx.scatter(diodeV[i-1], current_data[i-1], marker='.', s=20, edgecolor='black')
        bx.plot(voltage_fit, current_fit, color=colors[i-1], label=f'{name} diode {i}')

        mpld3.save_html(fig, "diode_graphs/inputOutputVoltage.html")
        mpld3.save_html(fig2, "diode_graphs/IVCurve.html")
        ain.stopAcquisition()
        libm2k.contextClose(adam)

    # Finalize and save plots
    print(f"Length of diodeV's values {len(diodeV[0])}, {len(diodeV[1])}, {len(diodeV[2])}")
    print(f"Length of current_data's values {len(current_data[0])}, {len(current_data[1])}, {len(current_data[2])}")
    averageDiodeV = [sum(x)/len(x) for x in zip(*diodeV)]
    averageCurrent = [sum(x)/len(x) for x in zip(*current_data)]
    bx.scatter(averageDiodeV, averageCurrent, marker = '.', s=20, edgecolor = 'black')
    param, _ = curve_fit(f=Exp, xdata=averageDiodeV, ydata=averageCurrent, p0=(0, 1, 0.1))
    I0, Vt, V_off = param
    current_fit = np.linspace(0, 0.05, 100)
    voltage_fit = (np.log((current_fit - V_off) / I0) / Vt)
    bx.plot(voltage_fit, current_fit, color = "orange", label = "Average Voltage across Diode (V)")

    ax.grid()
    ax.set_xlabel('Time(s)')
    ax.set_ylim([0, 5])
    ax.set_ylabel("Voltage")
    ax.legend()
    fig.savefig('vtGraph.png')

    bx.grid()
    bx.set_xlabel("Voltage Across Diode (V)")
    bx.set_ylim([0, 0.05])
    bx.set_ylabel("Input Current (A)")
    bx.legend()
    fig2.savefig(f'diode_graphs/{name}.png')
    plt.show()

diodeName = str(input("Name of Diode Being tested?: "))
CurveTrace(diodeName)
