import libm2k
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def CurveTrace(name):
    fig, ax = plt.subplots()
    fig2, bx = plt.subplots()

    diode_voltages = []
    diode_currents = []
    colors = ['red', 'blue', 'green']

    for i in range(1, 4, 1):
        halt = str(input(f"Please configure diode {i} for tracing. Press 'y' and 'enter' when ready..."))
        if halt != 'y':
            exit(1)

        # Initialize the device
        uri = libm2k.getAllContexts()
        if uri is None:
            print("Connection Error: No ADALM2000 device available/connected to your PC.")
            exit(1)
        else:
            print("Successfully connected ADALM2000...")
        
        adam = libm2k.m2kOpen(uri[0])
        adam.reset()
        adam.calibrate()
        adam.calibrateDAC()
        print("Calibration successful!")

        # Define the signal parameters
        duration = 6  # Duration in seconds
        scope_sample_rate = 1000  # Sample rate of the oscilloscope
        signal_sample_rate = 1000  # Total samples per second for output signal
        num_output_samples = duration * 1000 

        aout = adam.getAnalogOut()
        aout.enableChannel(0, True)
        t = np.linspace(0, duration, signal_sample_rate * duration)
        v_signal = 0.8 * t  # Input signal equation
        v_signal = np.clip(v_signal, -5, 5)
        aout.setCyclic(False)
        aout.setSampleRate(0, signal_sample_rate)
        aout.push(0, v_signal.tolist())

        # Oscilloscope setup
        ain = adam.getAnalogIn()
        ain.setSampleRate(scope_sample_rate)
        ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_1, True)
        ain.setRange(libm2k.ANALOG_IN_CHANNEL_1, 0, 5)
        ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_2, True)
        ain.setRange(libm2k.ANALOG_IN_CHANNEL_2, 0, 5)

        # Collect and process data
        diodeV, inputV = np.array(ain.getSamples(num_output_samples))
        inputV, indices1 = np.unique(inputV, return_index=True)
        diodeV = diodeV[indices1]
        time_x = np.linspace(0, duration, len(diodeV))

        # Plot voltage-time graph
        ax.plot(time_x, diodeV, label=f'Diode {i} Voltage (V)')
        ax.plot(time_x, inputV, label=f'Diode {i} Input Voltage (V)')
        
        # Calculate current and voltage for IV curve
        resistance = 100
        current_data = list((inputV - diodeV) / resistance for inputV, diodeV in zip(inputV, diodeV))
        current_data = np.array(current_data)
        diodeV, indices = np.unique(diodeV, return_index=True)
        current_data = current_data[indices]
        diodeV = np.clip(diodeV, a_min=0, a_max=1000000)

        # Handle duplicates by adding a small epsilon value
        unique_diodeV, unique_indices = np.unique(diodeV, return_index=True)
        if len(unique_diodeV) < len(diodeV):
            diodeV += np.random.uniform(0, 1e-5, size=diodeV.shape)

        # Interpolate data to common length and range
        common_length = 1000
        common_voltage_range = np.linspace(0, 0.5, common_length)  # Adjust this range as needed
        current_interp_func = interp1d(diodeV, current_data, kind='linear', fill_value='extrapolate')
        current_interp = current_interp_func(common_voltage_range)
        
        # Store interpolated data for averaging
        diode_voltages.append(common_voltage_range)
        diode_currents.append(current_interp)

        # Fit data to exponential model
        def Exp(V, I0, Vt, V_off):
            return I0 * np.exp(V * Vt) + V_off

        param, cov = curve_fit(f=Exp, xdata=diodeV, ydata=current_data, p0=(0, 1, 0.1))
        I0, Vt, V_off = param
        current_fit = np.linspace(0, 0.05, 100)
        voltage_fit = (np.log((current_fit - V_off) / I0) / Vt)

        # Plot IV curve
        bx.scatter(diodeV, current_data, marker='.', s=20, edgecolor='black')
        bx.plot(voltage_fit, current_fit, color=colors[i-1], label=f'{name} diode {i}')

        ain.stopAcquisition()
        libm2k.contextClose(adam)

    # Calculate and plot average IV curve
    avg_voltage = np.mean(diode_voltages, axis=0)
    avg_current = np.mean(diode_currents, axis=0)
    bx.plot(avg_voltage, avg_current, color='orange', label='Average Curve')

    # Finalize and save plots
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

CurveTrace("R0xB")
