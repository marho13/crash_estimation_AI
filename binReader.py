import numpy as np
import qd
import os
from scipy.signal import butter, lfilter, freqz
from qd.cae.dyna import Binout
import re
import matplotlib.pyplot as plt

class dataloader:
    def __init__(self, folderName):
        self.folderName = folderName
        self.getAllFiles()

    def getAllFiles(self):
        self.getAllVehicles()
        self.files, self.s = self.getAllSpeeds()
        print(self.files)

    def getAllSpeeds(self):
        speeds = []
        output = []
        folders = [(self.folderName + c) for c in self.cars]
        for f in folders:
            speeds.append([])
            s = os.listdir(f)
            if s != []:
                output.append(s)
            for x in s:
                speeds[-1].append(f+"/"+x+"/binout*")

        return speeds, output

    def getAllVehicles(self):
        self.cars = os.listdir(self.folderName)

    def butter_lowpass(self, cuttoff, fs, order=6):
        return butter(order, cuttoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=6):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def makeData(self, fileLoc): #Reads the data, gets the right sensor, change from mm to m and lastly filter the data
        index = 0
        for i in range(len(self.files)):
            for j in range(len(self.files[i])):
                if self.files[i][j] == fileLoc:
                    index = j

        binoutFile = Binout(fileLoc)
        #data = binoutFile.read('nodout', 'x_velocity')
        #data = binoutFile.read('nodout', 'x_acceleration')
        data = binoutFile.read('nodout', 'x_displacement')

        if "ADAS" in fileLoc:
            car = 1
            data, speed = self.differentCars(data, 0, index)

        elif "Chevy" in fileLoc:
            car = 2
            data, speed = self.differentCars(data, 1, index)

        else:
            car = 0
            data, speed = self.differentCars(data, 2, index)

        data = data / 1000 #   Deformation
        #data = data / 36000 #/ 100000  # mm-> m (10000 for g) Speed
        #data = data / 1000000 #Acceleration# mm-> m (10000 for g)
        data = self.butter_lowpass_filter(data, 3.667, fs=180, order=6) #Formatted data, filter has been passed over it
        return data, speed, car

    def differentCars(self, data, car, i):
        #print(i, car)
        stringy = self.s[car][i]
        speed = re.findall(r'\d+', stringy)[0]
        if car == 0:
            data = data[:, -1]
        elif car == 1:
            data = data[:, 0]
        else:
            data = data[:, 82]
        return data, speed


