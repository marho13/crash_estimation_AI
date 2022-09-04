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
            s = os.listdir(f)
            if s != []:
                output.append(s)
            for x in s:
                speeds.append(f+"/"+x+"/binout*")

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
        # print(fileLoc)
        index = 0
        for i in range(len(self.files)):
            if self.files[i] == fileLoc:
                index = i

        binoutFile = Binout(fileLoc)
        # print(binoutFile.read())
        car = 0
        stringy = self.s[0][index]
        speed = re.findall(r'\d+', stringy)[0]
        data = binoutFile.read('nodout', 'x_acceleration')
        data = data[:, 82]  # Specific sensor chosen by bobosan
        data = data / 10000  # mm-> m (10000 for g)
        data = self.butter_lowpass_filter(data, 3.667, fs=180, order=6) #Formatted data, filter has been passed over it
        return data, speed, car


