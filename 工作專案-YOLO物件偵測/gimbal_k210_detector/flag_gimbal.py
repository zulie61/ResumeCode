import math
import time
from machine import Timer,PWM
from fpioa_manager import *
from Maix import GPIO

def Servo(servo,angle):
    if 180 >= angle >= 0:
        servo.duty((angle/90 + 0.5)*(100/20))

class gimbal():
    def __init__(self,pin1,pin2):
        self.tim0 = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
        self.tim1 = Timer(Timer.TIMER1, Timer.CHANNEL1, mode=Timer.MODE_PWM)
        self.pin1 = pin1
        self.pin2 = pin2

    def correction(self): #雷射點對與平面邊緣中心點校正
        servoPWM1 = PWM(self.tim0, freq=50, duty=0, pin=self.pin1)
        servoPWM2 = PWM(self.tim1, freq=50, duty=0, pin=self.pin2)
        Servo(servoPWM1,3)  # 垂直
        Servo(servoPWM2,97) # 水平
        time.sleep(1)

    def angle(self,theta,phi):
        servoPWM1 = PWM(self.tim0, freq=50, duty=0, pin=self.pin1)
        servoPWM2 = PWM(self.tim1, freq=50, duty=0, pin=self.pin2)
        Servo(servoPWM1,theta)
        Servo(servoPWM2,phi)
        time.sleep(1)

    def coor2angle(self,x,y):
        z=-20
        x = 0.0012*x**3 - 0.034*x**2 + 1.1091*x +0.14125
        if x < 0:x=0

        orig = (0,0,0) # 假設原點為 servo 交接處

        # 4.2 為垂直軸心與猜想圓心距離、17 為猜想圓心與紙距離
        d = 4.2*math.tan(math.pi/2 - math.atan((z - 4.2)/20)) # radius
        sec_orig = (0,0,d)   # 更新原點
        coor     = (x,y,z)   # 目標點
        sec_coor = (x,y,z-d) # 目標點新座標

        # Cartesian coordinate system  =>  spherical coordinate system
        r     = math.sqrt(sec_coor[0]**2 + sec_coor[1]**2 + sec_coor[2]**2)
        theta = math.acos(-(sec_coor[0])/r)         # radius
        phi   = math.atan(sec_coor[1]/sec_coor[2])  # radius

        theta *= 180/math.pi # spherical degree
        phi   *= 180/math.pi # spherical degree

        theta  = theta-90 # hardware degree
        phi    = 97+phi  # hardware degree

        servoPWM1 = PWM(self.tim0, freq=50, duty=0, pin=self.pin1)
        servoPWM2 = PWM(self.tim1, freq=50, duty=0, pin=self.pin2)
        Servo(servoPWM1,theta+3)
        Servo(servoPWM2,phi)
        time.sleep(1)
    
    def deinit(self):
        servoPWM1.deinit()
        servoPWM2.deinit()


