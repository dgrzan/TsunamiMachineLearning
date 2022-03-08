#!/usr/bin/env python

import matplotlib.pyplot as plt
import csv
import numpy as np



if __name__ == '__main__':

    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data50nodes.txt', "r") as csvfile:
        data50 = list(csv.reader(csvfile))
    d50 = np.asarray(data50)
    d50 = d50.astype(np.float)
    
    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data100nodes.txt', "r") as csvfile:
        data100 = list(csv.reader(csvfile))
    d100 = np.asarray(data100)
    d100 = d100.astype(np.float)

    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data200nodes.txt', "r") as csvfile:
        data200 = list(csv.reader(csvfile))
    d200 = np.asarray(data200)
    d200 = d200.astype(np.float)

    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data400nodes.txt', "r") as csvfile:
        data400 = list(csv.reader(csvfile))
    d400 = np.asarray(data400)
    d400 = d400.astype(np.float)

    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data800nodes.txt', "r") as csvfile:
        data800 = list(csv.reader(csvfile))
    d800 = np.asarray(data800)
    d800 = d800.astype(np.float)
    
    samples50 = d50[:,1]
    error50 = d50[:,5]
    l1error50 = d50[:,4]
    sd50 = d50[:,7]
    dt50 = d50[:,8]
    l1terror50 = d50[:,2]

    samples100 = d100[:,1]
    error100 = d100[:,5]
    l1error100 = d100[:,4]
    sd100 = d100[:,7]
    dt100 = d100[:,8]
    l1terror100 = d100[:,2]

    samples200 = d200[:,1]
    error200 = d200[:,5]
    l1error200 = d200[:,4]
    sd200 = d200[:,7]
    dt200 = d200[:,8]
    l1terror200 = d200[:,2]

    samples400 = d400[:,1]
    error400 = d400[:,5]
    l1error400 = d400[:,4]
    sd400 = d400[:,7]
    dt400 = d400[:,8]
    l1terror400 = d400[:,2]

    samples800 = d800[:,1]
    error800 = d800[:,5]
    l1error800 = d800[:,4]
    sd800 = d800[:,7]
    dt800 = d800[:,8]
    l1terror800 = d800[:,2]

    
    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data100nodes01.txt', "r") as csvfile:
        data10001 = list(csv.reader(csvfile))
    d10001 = np.asarray(data10001)
    d10001 = d10001.astype(np.float)

    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data400nodes01.txt', "r") as csvfile:
        data40001 = list(csv.reader(csvfile))
    d40001 = np.asarray(data40001)
    d40001 = d40001.astype(np.float)
    
    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data50nodes01.txt', "r") as csvfile:
        data5001 = list(csv.reader(csvfile))
    d5001 = np.asarray(data5001)
    d5001 = d5001.astype(np.float)

    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/data200nodes01.txt', "r") as csvfile:
        data20001 = list(csv.reader(csvfile))
    d20001 = np.asarray(data20001)
    d20001 = d20001.astype(np.float)

    samples10001 = d10001[:,1]
    error10001 = d10001[:,5]
    l1error10001 = d10001[:,4]
    sd10001 = d10001[:,7]
    dt10001 = d10001[:,8]
    l1terror10001 = d10001[:,2]

    samples40001 = d40001[:,1]
    error40001 = d40001[:,5]
    l1error40001 = d40001[:,4]
    sd40001 = d40001[:,7]
    dt40001 = d40001[:,8]
    l1terror40001 = d40001[:,2]
    
    samples5001 = d5001[:,1]
    error5001 = d5001[:,5]
    l1error5001 = d5001[:,4]
    sd5001 = d5001[:,7]
    dt5001 = d5001[:,8]
    l1terror5001 = d5001[:,2]

    samples20001 = d20001[:,1]
    error20001 = d20001[:,5]
    l1error20001 = d20001[:,4]
    sd20001 = d20001[:,7]
    dt20001 = d20001[:,8]
    l1terror20001 = d20001[:,2]

    
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    fig, ax3 = plt.subplots()
    fig, ax4 = plt.subplots()
    fig, ax5 = plt.subplots()
    fig, ax6 = plt.subplots()
    fig, ax7 = plt.subplots()
    fig, ax8 = plt.subplots()
    fig, ax9 = plt.subplots()

    #plt.errorbar(samples400, error400, yerr=sd400, fmt='-o')
    #plt.errorbar(samples50, error50, yerr=sd50, fmt='-o')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~ overfitting ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ax4.plot(samples50, l1error50/l1terror50, '-o', color='lightsteelblue')
    ax4.plot(samples100, l1error100/l1terror100, '-o', color='cornflowerblue')
    ax4.plot(samples200, l1error200/l1terror200, '-o', color='blue')
    ax4.plot(samples400, l1error400/l1terror400, '-o', color='navy')
    ax4.set_ylim([0,5])
    ax4.set_ylabel("training error to testing error ratio")
    ax4.set_xlabel("input points")
    
    ax5.plot(samples5001, l1error5001/l1terror5001, '-o', color='pink')
    ax5.plot(samples10001, l1error10001/l1terror10001, '-o', color='lightcoral')
    ax5.plot(samples20001, l1error20001/l1terror20001, '-o', color='firebrick')
    ax5.plot(samples40001, l1error40001/l1terror40001, '-o', color='darkred')
    ax5.set_ylim([0,5])
    ax5.set_ylabel("training error to testing error ratio")
    ax5.set_xlabel("input points")

    ax6.plot(samples50, l1error50/l1terror50, '-o', color='lightsteelblue')
    ax6.plot(samples100, l1error100/l1terror100, '-o', color='cornflowerblue')
    ax6.plot(samples200, l1error200/l1terror200, '-o', color='blue')
    ax6.plot(samples400, l1error400/l1terror400, '-o', color='navy')
    ax6.plot(samples5001, l1error5001/l1terror5001, '-o', color='pink')
    ax6.plot(samples10001, l1error10001/l1terror10001, '-o', color='lightcoral')
    ax6.plot(samples20001, l1error20001/l1terror20001, '-o', color='firebrick')
    ax6.plot(samples40001, l1error40001/l1terror40001, '-o', color='darkred')
    ax6.set_ylim([0,5])
    ax6.set_ylabel("training error to testing error ratio")
    ax6.set_xlabel("input points")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~ error ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    ax1.plot(samples50, error50, '-o', color='lightsteelblue')
    ax1.plot(samples100, error100, '-o', color='cornflowerblue')
    ax1.plot(samples200, error200, '-o', color='blue')
    ax1.plot(samples400, error400, '-o', color='navy')
    ax1.plot(samples800, error800, '-o', color='black')
    ax1.set_ylim([0,2.2])
    ax1.set_ylabel("avg error (m)")
    ax1.set_xlabel("input points")

    ax2.plot(samples5001, error5001, '-o', color='pink')
    ax2.plot(samples10001, error10001, '-o', color='lightcoral')
    ax2.plot(samples20001, error20001, '-o', color='firebrick')
    ax2.plot(samples40001, error40001, '-o', color='darkred')
    ax2.set_ylim([0,2.2])
    ax2.set_ylabel("avg error (m)")
    ax2.set_xlabel("input points")

    ax3.plot(samples50, error50, '-o', color='lightsteelblue')
    ax3.plot(samples100, error100, '-o', color='cornflowerblue')
    ax3.plot(samples200, error200, '-o', color='blue')
    ax3.plot(samples400, error400, '-o', color='navy')
    ax3.plot(samples5001, error5001, '-o', color='pink')
    ax3.plot(samples10001, error10001, '-o', color='lightcoral')
    ax3.plot(samples20001, error20001, '-o', color='firebrick')
    ax3.plot(samples40001, error40001, '-o', color='darkred')
    ax3.set_ylim([0,2.2])
    ax3.set_ylabel("avg error (m)")
    ax3.set_xlabel("input points")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~ diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ax7.plot(samples50, dt50, '-o', color='lightsteelblue')
    ax7.plot(samples100, dt100, '-o', color='cornflowerblue')
    ax7.plot(samples200, dt200, '-o', color='blue')
    ax7.plot(samples400, dt400, '-o', color='navy')
    ax7.set_ylim([0,100])
    ax7.set_ylabel("Difference in inundated squares")
    ax7.set_xlabel("input points")

    ax8.plot(samples5001, dt5001, '-o', color='pink')
    ax8.plot(samples10001, dt10001, '-o', color='lightcoral')
    ax8.plot(samples20001, dt20001, '-o', color='firebrick')
    ax8.plot(samples40001, dt40001, '-o', color='darkred')
    ax8.set_ylim([0,100])
    ax8.set_ylabel("Difference in inundated squares")
    ax8.set_xlabel("input points")

    ax9.plot(samples50, dt50, '-o', color='lightsteelblue')
    ax9.plot(samples100, dt100, '-o', color='cornflowerblue')
    ax9.plot(samples200, dt200, '-o', color='blue')
    ax9.plot(samples400, dt400, '-o', color='navy')
    ax9.plot(samples5001, dt5001, '-o', color='pink')
    ax9.plot(samples10001, dt10001, '-o', color='lightcoral')
    ax9.plot(samples20001, dt20001, '-o', color='firebrick')
    ax9.plot(samples40001, dt40001, '-o', color='darkred')
    ax9.set_ylim([0,100])
    ax9.set_ylabel("Difference in inundated squares")
    ax9.set_xlabel("input points")
    
    plt.show()
