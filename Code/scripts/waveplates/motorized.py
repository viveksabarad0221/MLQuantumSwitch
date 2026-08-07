#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Get power reading from Thorlabs PM100D"""
"""To characterize waveplates or polarizers"""
"""In motorized rotation mount (PCBMotors or Thorlabs Elliptec)"""

import elliptec
from elliptec import ReportedError
import argparse
from ThorlabsPM100 import ThorlabsPM100, USBTMC
import pyvisa
import serial
import time
import csv
import numpy, scipy.optimize
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
import sys

OPTIC_LABELS = {'hwp': 'Half-wave plate', 'qwp': 'Quarter-wave plate', 'polarizer': 'Polarizer'}

parser = argparse.ArgumentParser(description='Characterize waveplates or polarizers in a motorized rotation mount (Thorlabs elliptec or PCBMotors) with a PM100D powermeter.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--waveplate-name' , type=str, help='String to identify optic under test', default="")
parser.add_argument('-p', '--powermeter-device' , type=str, help='path to PM100D usbtmc device', default='') # Only for Linux
parser.add_argument('-d', '--motor-device' , type=str, help='path to rotation mount serial device', default='') #"COM9" HWP1:COM8, HWP2:COM10 HWP3:COM3

parser.add_argument('-m', '--motor-number' , type=str, help='rotation mount motor number', default='') #0
parser.add_argument('-n', '--serial-number' , type=str, help='Serial number of powermeter (use on windows)', default='') #P0049446
#parser.add_argument('-s', '--save-plot' , type=bool, action=argparse.BooleanOptionalAction, help='save characterization plot', default=True)
#parser.add_argument('-s', '--save-plot' , type=bool, action='store_true', help='save characterization plot', default=True)
parser.add_argument('-t', '--motor-type' , type=str, help="Motorized rotation mount type. Either 'elliptec' or 'pcbmotor'", default='elliptec')
parser.add_argument('-y', '--optic-type' , type=str, choices=['hwp', 'qwp', 'polarizer'], help="Type of optic under test: 'hwp', 'qwp' or 'polarizer'. Sets the expected modulation period for the sinusoidal fit.", default='hwp')
args = parser.parse_args()
saveplot=True

if sys.platform != "linux":
    WINDOWS=True
    import pyvisa as visa
    eol='\r\n'
    if args.serial_number == "":
        print("Please provide a powermeter serial number")
        sys.exit()
    else:
        sn = args.serial_number
else:
    WINDOWS=False
    eol='\r'
    if args.powermeter_device == "":
        print("Please provide a powermeter device path")
        sys.exit()
    else:
        pmdevice = args.powermeter_device

# get file/waveplate name
if args.waveplate_name == "" :
    askstr="Please enter a waveplate name: "
    wpname=input(askstr)
    if len(wpname)==0:
        wpname="wp"
    wpname=wpname.replace(" ","_")
else:
    wpname=args.waveplate_name.replace(" ","_")

if args.motor_type == 'elliptec':
    motortype = 'elliptec'
    if args.motor_number == '':
        print('please specify motor number to characterize')
        sys.exit()
    else:
        mnumstr=args.motor_number
        ELLIPTEC_ALL_ADDR=[mnumstr]
elif args.motor_type == 'pcbmotor':
    motortype = 'pcb'
    if args.motor_number == '':
        print('please specify motor number to characterize')
        sys.exit()
    else:
        mnumstr=args.motor_number
else:
    print('Please specify either elliptec or pcbmotor as motor type')
    sys.exit()


if args.motor_device == '':
    print('Please specify motor device path')
    sys.exit()
else:
    pcbdevice = args.motor_device


# fit functions
def sinfunc(param,x):
    return param[0]+param[1]*numpy.sin(param[2]*x+param[3])
def sinfuncerr(param,x,y):
    return sinfunc(param,x)-y

def motor_home(c, m):
    #c: controller (serial.Serial)
    #m: motor (str)
    cmd=m+',s-2880'+eol
    print(cmd)
    c.write(cmd.encode())
    while(1):
        line=c.readline()
        print(line)
        if ('Steps' in str(line)):
            print('steps string detected')
            break
        else:
            time.sleep(0.05)

def motor_move(c, m, steps):
    #c: controller (serial.Serial)
    #m: motor (str)
    #steps: number of steps to go
    cmd=m+',s'+str(int(steps))+eol
    print(cmd)
    c.write(cmd.encode())
    while(1):
        line=c.readline()
        print(line)
        if ('Steps' in str(line)):
            break
        else:
            time.sleep(0.05)

def motor_print_cw(c,m):
    cmd=m+',cw'+eol
    print(cmd)
    c.write(cmd.encode())
    time.sleep(0.05)
    while c.in_waiting:
        line=c.readline()
        print(line)
    cmd=m+',ccw'+eol
    print(cmd)
    c.write(cmd.encode())
    time.sleep(0.05)
    while c.in_waiting:
        line=c.readline()
        print(line)

def controller_open(dev):
    if motortype=='pcb':
        #dev: device path (str)
        ser=serial.Serial(dev,19200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.2, xonxoff=False, rtscts=False, write_timeout=None, dsrdtr=False)
        #Python does not honour Flow contrl options during init. Wasted a f*ing day figuring this out
        ser.dtr=False
        ser.rts=False
        return ser
    elif motortype=='elliptec':
        m = elliptec.Elliptec(dev=dev, addrs=ELLIPTEC_ALL_ADDR, home=True, freq=True)
        return m

dev=controller_open(pcbdevice)

fname=wpname+".csv"
pi=numpy.pi

#open file
f=open(fname, mode='w', newline='\n')
fwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
fwriter.writerow(["ang",wpname])

angles=numpy.arange(0,321,1)
power=numpy.zeros(0)
header=[]
indat=[]

print(angles)
#connect to powermeter powermeter
if WINDOWS:
    rm = visa.ResourceManager()
    inst=rm.open_resource("USB0::0x1313::0x8078::"+sn+"::INSTR")
    inst.read_termination = '\n'
    inst.write_termination = '\n'
    inst.timeout = 1000
else:
    inst=USBTMC(device=pmdevice) # type: ignore
pm = ThorlabsPM100(inst=inst)
pm.sense.average.count=100
PuW = -1
PW = pm.read
PuW = PW*1000000



if motortype=='pcb':
    motor_home(dev,'m1')
    time.sleep(2)
    motor_home(dev,'m2')
    time.sleep(2)
    motor_home(dev,'m3')
    time.sleep(2)


    #get data...
    prevang=0
    for ang in angles:
        print(ang)
        motor_move(dev,mnumstr,(ang-prevang)*8)
        prevang=ang
        time.sleep(1.5)
        PW=pm.read*1000
        print(PW)
        power=numpy.append(power,PW)
        fwriter.writerow([ang,PW])
    f.close()
    if WINDOWS:
        rm.close()

elif motortype=='elliptec':
    dev.home(mnumstr)

    #get data...
    prevang=0
    for ang in angles:
        print(ang)
        time.sleep(0.5)
        try:
            print(dev.moveabsolute(mnumstr,ang))
        except ReportedError as e:
            print(f"Caught sensor error: {e}")
            dev.home(mnumstr)
            dev.moveabsolute(mnumstr, ang)
            print("second time failed")


        prevang=ang
        time.sleep(0.1)
        PW=pm.read*1000
        print(PW)
        power=numpy.append(power,PW)
        fwriter.writerow([ang,PW])
    f.close()
    if WINDOWS:
        rm.close()


with open(fname, 'r', newline='\n') as infile:
    tmpreader = csv.reader(infile, delimiter=',')
    for row in tmpreader:
        indat.append(row)

header=indat[0][1:]
angles=numpy.transpose(indat)[0][1:]
angles2=indat[1:][0]

wp=[]

for i in range(1,len(indat)):
    tmp=[]
    for j in range(1,len(indat[i])):
        tmp.append(float(indat[i][j]))
    wp.append(tmp)

header=indat[0][1:]

angles=[]
for i in range(1,len(indat)):
    angles.append(float(indat[i][0]))

wp=numpy.array(wp).T.tolist()
angles=numpy.asarray(angles)

print(len(wp))
plotrange=numpy.arange(0,320,0.001)

# HWP/QWP under test (between a fixed input polarizer and a fixed analyzer) both
# modulate the power with a 90 deg period (cos(4*theta) term) regardless of
# retardance - only their contrast/offset differs. A bare polarizer under test
# follows Malus's law instead, with a 180 deg period - half the frequency.
freq_guess = 1/(8*pi) if args.optic_type == 'polarizer' else 1/(4*pi)

for i in range(0,len(wp)):
    #fit
    parest=[1.,max(wp[i]),freq_guess,0.]

    par,success=scipy.optimize.leastsq(sinfuncerr, parest[:], args=(angles,wp[i]))

    #plot
    fig, ax1 = plt.subplots()
    ax1.plot(angles,wp[i],linestyle='none',marker='o',ms=3)
    ax1.plot(plotrange,sinfunc(par,plotrange))
    plt.title(OPTIC_LABELS[args.optic_type] + ' ' + header[i], fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rotation angle [°]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Power, H polarized (a.u.)', fontsize=12, fontweight='bold')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    maximaindices=argrelmax(sinfunc(par,plotrange), order=20)
    maxtext='Maxima at '
    maxima=[]
    for j in maximaindices[0]:
        maxima.append(plotrange[j])
    for k in range(0,len(maxima)):
        maxtext=maxtext+'{0:.2f}°, '.format(maxima[k])
    maxtext=maxtext[:-2]
    #maxtext=maxtext+'. Frequency: {0:.3f}'.format(par[2])
    print(maxima)
    fig.tight_layout()
    ax1.text(0.01,0.01,maxtext, fontsize='8',color='black',transform=ax1.transAxes)
    #save to file
    if saveplot:
        plotsavename=wpname
        print('saving ' + plotsavename)
        plt.savefig(plotsavename+'.pdf', format="pdf",transparent=True, bbox_inches='tight', pad_inches=0)
        plt.savefig(plotsavename+'.png', format="png",transparent=True, bbox_inches='tight', pad_inches=0)
        plt.show()
