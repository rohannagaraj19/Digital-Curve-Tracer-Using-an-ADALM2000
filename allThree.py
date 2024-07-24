#created by Rohan Nagaraj
#for the test of diode operation
import libm2k #library for microcontroller access on ADALM2000
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpld3
import pandas as pd
import csv
import os
# Fit data to exponential model
def Exp(V, I0, Vt, V_off):
    return I0 * np.exp(V * Vt) + V_off
class Diode:
    def __init__(self, name, currentMax):
        self.fig, self.ax = plt.subplots() #input voltage and voltage response subplot
        self.fig2, self.bx = plt.subplots() #iv curve subplot
        self.colors = ['red', 'blue', 'green']
        self.diodeV = [] #contains voltage data across the diode ()
        self.current_data = []
        self.inputV = []
        self.newV = [] #contains voltage response of new diode
        self.newA = [] #contains current input of new diode
        self.averageV = []
        self.averageA = []
        self.name = name
        self.newFit = [] #stores the equation for the 'Average Voltage across new Diode'
        self.avgFit = [] #stores the equation for the 'Average Voltage across tested Diode'
        self.currMax = np.float64(currentMax) #max current it plots
        self.currSample = 1000 #precision of plot (currSample amount of evenly spaced points)
        self.newSig = True #a signal to make sure standard dev doesnt run without a new graph

    def save_graphs(self):
        name_ = self.name
        mpld3.save_html(self.fig, "diode_graphs/inputOutputVoltage.html")
        mpld3.save_html(self.fig2, "diode_graphs/IVCurve.html")
        self.fig.savefig('diode_graphs/vtGraph.png')
        self.fig2.savefig(f'diode_graphs/{name_}.png')
        plt.show()

    def CurveTrace(self):
        diodeV = []
        for i in range(3):
            diodeV.append([])
        self.diodeV = diodeV #sets these as class variables for access in other functions

        current_data = []
        for i in range(3):
            current_data.append([])
        self.current_data = current_data

        inputV = []
        for i in range(3):
            inputV.append([])
        self.inputV = inputV

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
            v_signal = 0.8*t  #input signal equation
            #v_signal = np.clip(v_signal, 0, 5)
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
            self.ax.plot(time_x, diodeV[i-1], label=f'Diode {i} Voltage (V)')
            self.ax.plot(time_x, inputV[i-1], label=f'Diode {i} Input Voltage (V)')
            
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
            current_fit = np.linspace(0.001, self.currMax, self.currSample)
            voltage_fit = (np.log((current_fit - V_off) / I0) / Vt)

            # Plot IV curve
            self.bx.scatter(diodeV[i-1], current_data[i-1], marker='.', s=20, edgecolor='black')
            self.bx.plot(voltage_fit, current_fit, color=self.colors[i-1], label=f'{self.name} diode {i}')

            
            ain.stopAcquisition()
            libm2k.contextClose(adam)

        # Finalize and save plots
        #print(f"Length of diodeV's values {len(diodeV[0])}, {len(diodeV[1])}, {len(diodeV[2])}")
        #print(f"Length of current_data's values {len(current_data[0])}, {len(current_data[1])}, {len(current_data[2])}")
        
        #find averages of voltage response and current input
        averageDiodeV = [sum(x)/len(x) for x in zip(*diodeV)]
        averageCurrent = [sum(x)/len(x) for x in zip(*current_data)]
        self.averageV = averageDiodeV #set pointers to the lists in the class variables
        self.averageA = averageCurrent

        self.bx.scatter(averageDiodeV, averageCurrent, marker = '.', s=20, edgecolor = 'black')
        param, _ = curve_fit(f=Exp, xdata=averageDiodeV, ydata=averageCurrent, p0=(0, 1, 0.1))
        I0, Vt, V_off = param
        current_fit = np.linspace(0.001, self.currMax, self.currSample)
        voltage_fit_avg = (np.log((current_fit - V_off) / I0) / Vt)
        self.avgFit = voltage_fit_avg #set the class pointer
        self.bx.plot(voltage_fit_avg, current_fit, color = "orange", label = "Average Voltage across Diode (V)")

        self.ax.grid()
        self.ax.set_xlabel('Time(s)')
        self.ax.set_ylim([0, 5])
        self.ax.set_ylabel("Voltage")
        self.ax.legend()

        self.bx.grid()
        self.bx.set_xlabel("Voltage Across Diode (V)")
        self.bx.set_ylim([0, self.currMax])
        self.bx.set_ylabel("Input Current (A)")
        self.bx.legend()

        #save averageDiodeV, averageCurrent, diodeV, current_data, inputV
        max_length = max(len(averageDiodeV), len(averageCurrent), len(diodeV[0]), 
                    len(diodeV[1]), len(diodeV[2]), len(current_data[0]), len(current_data[1]), 
                    len(current_data[2]), len(inputV[0]))
        averageDiodeV_ = averageDiodeV.copy()
        averageCurrent_ = averageCurrent.copy()
        averageDiodeV_ += [np.nan] * (max_length - len(averageDiodeV_)) #fills all values that arent indexed with nulls
        averageCurrent_ += [np.nan] * (max_length - len(averageCurrent_))
        for i in range(3):
            diodeV[i] = np.pad(diodeV[i], (0, max_length - len(diodeV[i])), constant_values=np.nan)
            current_data[i] = np.pad(current_data[i], (0, max_length - len(current_data[i])), constant_values=np.nan)
        inputV[0] = np.pad(inputV[0], (0, max_length - len(inputV[0])), constant_values=np.nan)
        rows = zip(averageDiodeV_, averageCurrent_, diodeV[0], diodeV[1], diodeV[2], current_data[0],
                current_data[1], current_data[2], inputV[0]) #groups all data so everything coincides with one another
        with open(f'diode_data/{self.name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Average Diode Voltage (V)', 'Average Current Through Diode (A)',
                            'Diode 1 Voltage(V)', 'Diode 2 Voltage(V)', 'Diode 3 Voltage(V)',
                            'Current Through Diode 1 (A)', 'Current Through Diode 2 (A)', 'Current Through Diode 3 (A)',
                            'Input Voltage'])
            writer.writerows(rows)
        #plt.show()


    def GraphNew(self, name):
        if name[0] == 'R' and name[3] == 'A':
            leName = 'R0xA'
        elif name[0] == 'R' and name[3] == 'B':
            leName = 'R0xB'
        elif os.path.exists(os.path.join('diode_data', name + " new.csv")):
            leName = name
        else:
            self.newSig = False
            return None
        reader = pd.read_csv(f'diode_data/{leName} new.csv') #dataframe
        yAxis = reader['Average Current Through Diode (A)'].copy().dropna()
        xAxis = reader['Average Diode Voltage (V)'].copy().dropna()
        self.newV = xAxis #set variables to class
        self.newA = yAxis
        param, _ = curve_fit(f=Exp, xdata=xAxis, ydata=yAxis, p0=(0, 1, 0.1))
        I0, Vt, V_off = param
        current_fit = np.linspace(0.001, self.currMax, self.currSample)
        voltage_fit_new = (np.log((current_fit - V_off) / I0) / Vt)
        self.bx.plot(voltage_fit_new, current_fit, color = "cyan", label = f"Average Voltage New {name} Diode (V)")
        self.newFit = voltage_fit_new
    def standardDev(self):
        if self.newSig == False:
            return None
        diodeV = self.averageV #set pointers to class variables (its annoying to keep typing self before every access of an array)
        current = self.averageA
        newV = self.newV
        newA = self.newA
        tolerance = 0.05 #5 percent tolerance
        above = 1/(1+tolerance)
        below = 1/(1-tolerance)

        
        above_tolerance_newV = list(above * x for x in newV)
        above_tolerance_newA = list(above * x for x in newA)
        below_tolerance_newV = list(below * x for x in newV) 
        below_tolerance_newA = list(below * x for x in newA)

        current_fit= np.linspace(0.001, self.currMax, self.currSample)
        param_abN, _ = curve_fit(Exp, xdata = above_tolerance_newV, ydata= above_tolerance_newA, p0 = (0,1,0.1))
        param_bN, _ = curve_fit(Exp, xdata = below_tolerance_newV, ydata = below_tolerance_newA, p0 = (0,1,0.1))

        voltage_fit_abN = (np.log((current_fit - param_abN[2]) / param_abN[0]) / param_abN[1]) #check if right indices
        voltage_fit_bN = (np.log((current_fit - param_bN[2]) / param_bN[0]) / param_bN[1])


        self.bx.plot(voltage_fit_abN, current_fit, color = 'cyan', linestyle = ':') #graph the tolerances
        self.bx.plot(voltage_fit_bN, current_fit, color = 'cyan', linestyle = ':')
        
        deviations = []
        for i in range(3): #self.avgFit
            if i ==0:
                differences = np.vstack((self.avgFit, voltage_fit_abN))
                deviations.append(np.std(differences))
            elif i==1:
                differences = np.vstack((self.avgFit, self.newFit))
                deviations.append(np.std(differences))
            elif i==2:
                differences = np.vstack((self.avgFit, voltage_fit_bN))
                deviations.append(np.std(differences))   

        print(f"This diode's deviation from the a new one is {(deviations)}")
        if(min(deviations) < 0.05): #if this deviation is less than 5% thats good enough 
            print(f"This diode is still operational")
        else:
            print(f"This diode is too worn out and not advised for operation")

print("To input new diode data, please state the diode name and type 'new' after")
diodeName = str(input("Name of Diode Being tested?: "))
plot_max = str(input("Please specify the range of current (A) you want to plot (0.05 Amps will show all the sample data, 10 Amps and above will predict data up to that range): "))
if(plot_max == ""):
    plot_max = 0.05
diode = Diode(diodeName, plot_max)
if 'new' in diodeName: #to add new data on a diode
    print("You are now entering the data for a new diode...")
    diode.CurveTrace()
    diode.save_graphs()
    exit(1)

diode.GraphNew(diodeName)
diode.CurveTrace()
diode.standardDev()
diode.save_graphs()

