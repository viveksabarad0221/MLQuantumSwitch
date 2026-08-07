#!/usr/bin/env python3

#U
#for linux:
# devpath='/dev/serial/by-id/usb-FTDI_FT230X_Basic_UART_DK0ENDSE-if00-port0' 
#for mac:
#devpath='/dev/tty.usbserial-DK0BKCL1'
#for windows:
devpath='COM3'


#mnumstr='0'

print("HELLO dear user, please put the motor number in (for eg): '0', '1', '2', ")
mnumstr = input()
print("insert the reference angle")
input_a = float(input())
# print("do you want +(p) or -(m) 45° to angle? (p/m)")
# input_b = input()

# if input_b == "p":
#     angle=input_a + 45
# if input_b == "m":
#     angle=input_a - 45

print("Enter the angle adjustment (+/- in degrees): ")
angle_adjustment = float(input())

angle = input_a + angle_adjustment

if angle >= 360:
    angle = angle-360
if angle <= 0:
    angle = angle+360

print("full angle is: ", angle)
print("now wait")
import elliptec


m= elliptec.Elliptec(devpath, [mnumstr], home=True, freq=True)
m.moveabsolute(mnumstr, angle)
print("ended")
