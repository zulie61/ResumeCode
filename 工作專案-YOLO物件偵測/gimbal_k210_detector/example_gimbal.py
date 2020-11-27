from flag_gimbal import gimbal

gimbal = gimbal(15,7)

gimbal.correction()      #校正

def control_angle():     #控制角度
    while True:
        theta,phi = eval(input('Enter angle:'))
        gimbal.angle(theta,phi)

def control_coordinate(): #坐標系轉角度
    while True:
        x,y = eval(input('Enter coordinates:'))
        gimbal.coor2angle(x,y)

if __name__ == '__main__':
    #control_angle()
    control_coordinate()
