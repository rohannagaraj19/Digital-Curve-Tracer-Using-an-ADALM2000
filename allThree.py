#created by Rohan Nagaraj
#for the test of diode operation
import libm2k #library for microcontroller access on ADALM2000
import numpy as np #linear algebra library
import matplotlib.pyplot as plt #graph plotting library
from scipy.optimize import curve_fit #curve fit library
import mpld3 #will import the most recent graphs to an interactive html
import pandas as pd #this is how we read and write csv files 
import csv #this is how we save csv files
import os #this is how we access some folders and files

# Fit data to exponential model
def Exp(V, I0, Vt, V_off):
    return I0 * np.exp(V * Vt) + V_off #our curves have an equation of this form... will return another array with each value corresponding to the equations t value
class Diode:
    def __init__(self, name, currentMax, resistance_): #contructor... contains all the class variables
        self.fig, self.ax = plt.subplots() #input voltage and voltage response subplot
        self.fig2, self.bx = plt.subplots() #iv curve subplot
        self.colors = ['red', 'blue', 'green'] #colors used to plot diode 1, 2, and 3
        self.diodeV = [] #contains voltage data across the diode ()
        self.current_data = [] #contains the input current data
        self.inputV = [] #contains the measured input voltage data
        self.newV = [] #contains voltage response of new diode
        self.newA = [] #contains current input of new diode
        self.averageV = [] #contains the average value of the three recorded voltages
        self.averageA = [] #same for current ^
        self.name = name #name of the diode being graphed
        self.newFitV = [] #stores the equation for the 'Average Voltage across new Diode'
        self.avgFitV = [] #stores the equation for the 'Average Voltage across tested Diode'
        self.currMax = np.float64(currentMax) #max current it plots
        self.currSample = 1000 #precision of plot (currSample amount of evenly spaced points)
        self.newSig = True #a signal to make sure standard dev doesnt run without a new graph
        self.resistance = resistance_      #what resistance is in series with the diode
        self.tolerance = 0.5 #5 percent tolerance

    def save_graphs(self): #called at the end of the program... saves all the graphs and data in a folder
        name_ = self.name
        mpld3.save_html(self.fig, "diode_graphs/inputOutputVoltage.html") #saves the graph as an interactive plot (will overwrite every time you run the program)
        mpld3.save_html(self.fig2, "diode_graphs/IVCurve.html")
        self.fig.savefig('diode_graphs/vtGraph.png') #saves the graph as a png (this is the voltage vs time graph)
        self.fig2.savefig(f'diode_graphs/{name_}.png') #the current vs voltage graph
        plt.show()

    def CurveTrace(self): #this is the only function that uses the ADALM2000 hardware and circuit
        diodeV = []
        for i in range(3): #i do this so diodeV can be a 3x1 array... ex: [[1,2,4], [6,3,1], [9,4,8]]
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
            if not(halt == "y" or halt == "" or ("yes" in halt) or ("Yes" in halt) or halt == "Y"): 
                exit(1) #quits program

            # Initialize the device
            uri = libm2k.getAllContexts() #an array of all the ip addresses available
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
            scope_sample_rate = 1000  #sample rate of the oscilloscope (max is 1000)
            signal_sample_rate = 75000 # Total samples per second for output signal (must surpass Nyquist Sample rate to prevent aliasing)
            num_output_samples = duration * scope_sample_rate 

            aout = adam.getAnalogOut() #an object that has all the functions for the signal generator
            aout.enableChannel(0, True) #enables channel 1 (indexed at 0)... if it said aout.enableChannel(1,True) then it turned on channel 2
            t = np.linspace(0, duration, signal_sample_rate * duration) #evenly spaced array of float numbers that we use as a time input
            
            v_signal = 4*t  #input signal equation
            v_signal = np.clip(v_signal, 0, 5) #will clip off the values in the arrays that are less than 0 and greater than 5
            
            aout.setCyclic(False) #makes it so the signal doesn't repeat
            aout.setSampleRate(0, signal_sample_rate) #param: channel (0 means channel 1), sampling frequency of input signal (typically 1000 Hz but needs to be greater than the Nyquist rate)
            aout.push(0, v_signal.tolist()) #this is how we tell the signal generator what voltages to output

            # Oscilloscope setup
            ain = adam.getAnalogIn() #gives us an object that contains all the functions for the oscilloscope
            ain.setSampleRate(scope_sample_rate)  # sample rate
            # Initialize analog input (oscilloscope) for Channel 1
            ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_1, True)  # Channel 1 (the orange cables)
            ain.setRange(libm2k.ANALOG_IN_CHANNEL_1, 0, 5)  # Channel 1 range (measures from 0 to 5 V)

            # Initialize analog input (oscilloscope) for Channel 2
            ain.enableChannel(libm2k.ANALOG_IN_CHANNEL_2, True)  # Channel 2 (the blue cables)
            ain.setRange(libm2k.ANALOG_IN_CHANNEL_2, 0, 5)  # Channel 2 range

            # Collect and process data
            diodeV[i-1], inputV[i-1] = np.array(ain.getSamples(num_output_samples)) #diodeV has all the channel 1 data and inputV has all the channel 2 data
            inputV[i-1], indices1 = np.unique(inputV[i-1], return_index=True) #delete all duplicate values (we are getting the indices that were kept from the original)
            diodeV[i-1] = diodeV[i-1][indices1] #this is a continuation of deleting all dup values
            # Time characteristics
            time_x = np.linspace(0, duration, len(diodeV[i-1]))

            # Plot voltage-time graph
            self.ax.plot(time_x, diodeV[i-1], label=f'Diode {i} Voltage (V)')
            self.ax.plot(time_x, inputV[i-1], label=f'Diode {i} Input Voltage (V)')
            
            # Calculate current and voltage for IV curve
            resistance = self.resistance

            current_data[i-1] = list((inputV - diodeV) / resistance for inputV, diodeV in zip(inputV[i-1], diodeV[i-1])) #basic KVL formula
            current_data[i-1] = np.array(current_data[i-1]) #convert to an numpy array
            # Delete duplicate values of current now
            diodeV[i-1], indices = np.unique(diodeV[i-1], return_index=True)
            current_data[i-1] = current_data[i-1][indices]
            diodeV[i-1] = np.clip(diodeV[i-1], a_min=0, a_max=10e6) #we throw away all the negative values



            #curve fitting
            param, _ = curve_fit(f=Exp, xdata=diodeV[i-1], ydata=current_data[i-1], p0=(0, 1, 0.1)) #default p0 that stores the initial guess of all the coefficients of the exponential function
            #param, cov = curve_fit(f=Exp, xdata=diodeV[i-1], ydata=current_data[i-1], p0=(0, 1, 0.1)) #in case you want to use the covariance matrix to understand how well the curve is fit
            I0, Vt, V_off = param
            current_fit = np.linspace(0.001, self.currMax, self.currSample) 
            voltage_fit = (np.log((current_fit - V_off) / I0) / Vt) #this is our v(current) equation... we flip the exponential function so it graphs nicely...

            # Plot IV curve
            self.bx.scatter(diodeV[i-1], current_data[i-1], marker='.', s=20, edgecolor='black')
            self.bx.plot(voltage_fit, current_fit, color=self.colors[i-1], label=f'{self.name} diode {i}') #notice we graph it flipped again so its exponential

            
            ain.stopAcquisition() #stops signal generator and oscillscope
            libm2k.contextClose(adam) #stops reading and writing data here from ADALM2000
        
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
        self.avgFitV = voltage_fit_avg #set the class pointer
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
        


    def GraphNew(self, name):
        if name[0] == 'R' and name[3] == 'A':
            leName = 'R0xA'
        elif name[0] == 'R' and name[3] == 'B':
            leName = 'R0xB'
        elif os.path.exists(os.path.join('diode_data', name + " new.csv")):
            leName = name #only works for non R0xA and R0xB diodes... needs the 
        else:
            self.newSig = False
            return None
        reader = pd.read_csv(f'diode_data/{leName} new.csv') #dataframe
        yAxis = reader['Average Current Through Diode (A)'].copy().dropna() #removes all the null values from the csv file and puts the actual data in an array
        xAxis = reader['Average Diode Voltage (V)'].copy().dropna()
        self.newV = xAxis #set variables to class
        self.newA = yAxis
        param, _ = curve_fit(f=Exp, xdata=xAxis, ydata=yAxis, p0=(0, 1, 0.1))
        I0, Vt, V_off = param
        current_fit = np.linspace(0.001, self.currMax, self.currSample)
        voltage_fit_new = (np.log((current_fit - V_off) / I0) / Vt)
        self.bx.plot(voltage_fit_new, current_fit, color = "cyan", label = f"Average Voltage New {name} Diode (V)")
        self.newFitV = voltage_fit_new
    
    def percentInBand(self): 
        if self.newSig == False:
            return None
        newV = self.newV
        newA = self.newA
        tolerance = self.tolerance
        above = 1/(1+tolerance)
        below = 1/(1-tolerance)

        
        above_tolerance_newV = list(above * x for x in newV)
        above_tolerance_newA = list(above * x for x in newA)
        below_tolerance_newV = list(below * x for x in newV) 
        below_tolerance_newA = list(below * x for x in newA)

        current_fit= np.linspace(0.001, self.currMax, self.currSample)
        param_abN, _ = curve_fit(Exp, xdata = above_tolerance_newV, ydata= above_tolerance_newA, p0 = (0,1,0.1))
        I0_abn, Vt_abn, V_off_abn = param_abN
        param_bN, _ = curve_fit(Exp, xdata = below_tolerance_newV, ydata = below_tolerance_newA, p0 = (0,1,0.1))
        I0_bn, Vt_bn, V_off_bn = param_bN

        voltage_fit_abN = (np.log((current_fit - V_off_abn) / I0_abn) / Vt_abn) #this is in a form that is useful for plotting numbers
        voltage_fit_bN = (np.log((current_fit - V_off_bn) / I0_bn) / Vt_bn)


        self.bx.plot(voltage_fit_abN, current_fit, color = 'cyan', linestyle = ':') #graph the tolerances
        self.bx.plot(voltage_fit_bN, current_fit, color = 'cyan', linestyle = ':')

        def tolerance_percentage(master_curve, compare_curve):
            difference = np.abs(compare_curve - master_curve) #absolute difference between the two curves
            tolerance = self.tolerance * master_curve #we can get the tolerance band with this
            within_tolerance = difference <= tolerance #we create a boolean array to see if a point is within the tolerance
            percentage_within_tolerance = np.sum(within_tolerance) / len(within_tolerance) #now we just find the percentage of points that are "true" in the boolean array
            return percentage_within_tolerance
        p_ofTolerance = tolerance_percentage(self.newFitV, self.avgFitV)
        self.bx.text(x=0.25, y=0.135, s= f"P_ofTolerance: {p_ofTolerance*100}", fontdict= None) #this is hardcoded... the position needs to be dynamic with the x max and y max of the graph
        print(f"The percentage of the tested curve in the 5% band is: {100 * p_ofTolerance}")
        if(p_ofTolerance >= 0.9): #if 90% of the curve is within tolerance, the diode should be fine 
            print(f"This diode is still operational")
        else:
            print(f"This diode is too worn out and not advised for operation")


print("To input new diode data, please state the diode name and type 'new' after")
diodeName = str(input("Name of Diode Being tested?: "))
plot_max = str(input("Please specify the range of current (A) you want to plot (0.5 Amps will show all the sample data, 10 Amps and above will predict data up to that range): "))
if(plot_max == ""):
    plot_max = 0.5

resistance = 10.1 #THE RESISTOR USED IN THE CIRCUIT (VERY CRUCIAL THAT THIS IS UPDATED IF YOU CHANGE THE CIRCUIT)
diode = Diode(diodeName, plot_max, resistance)
if 'new' in diodeName: #to add new data on a diode
    print("You are now entering the data for a new diode...")
    diode.CurveTrace()
    diode.save_graphs()
    exit(1)

diode.GraphNew(diodeName)
diode.CurveTrace()
diode.percentInBand()
diode.save_graphs()

