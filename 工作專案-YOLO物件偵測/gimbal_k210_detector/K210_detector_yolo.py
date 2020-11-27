import sensor,image,lcd,time
import KPU as kpu
from flag_gimbal import gimbal
import gc
#軸心與紙面保持 17cm
gc.enable()
gc.threshold(3000)
gimbal = gimbal(15,7) #(垂直腳位,水平腳位)
gimbal.correction()
time.sleep(1)

lcd.init(freq=15000000) #lcd 螢幕 、sensor 相機
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(0)
#sensor.set_vflip(1)
#sensor.set_windowing((224,224))
sensor.set_brightness(1)
#sensor.set_contrast(-1)
#sensor.set_auto_gain(1,2)
sensor.run(1)

#統計
thresh = 10 #剔離群值
sampleTimes = 20

clock = time.clock()


classes = ["A_Liang"]
#task = kpu.load("/sd/flag_man.kmodel")
print('Model Loading...')
task = kpu.load(0x300000) #使用gui燒錄model
anchor = (1, 1.2, 2, 3, 4, 3, 6, 4, 5, 6.5)
kpu.init_yolo2(task, 0.17, 0.3, 5, anchor)
print('Initialized')
pdCount = 1   # 預測幾次
totalPos = [] # 存 20 次座標



while True:
    clock.tick()
    img = sensor.snapshot()
    img2 = img.resize(224,224)
    a=img2.pix_to_ai();

    code = kpu.run_yolo2(task, img2)
    #print(clock.fps())
    if code:
        for i in code:
            i1 = list(i.rect())
            i1[0]= int(i1[0]*320/224)
            i1[1]= int(i1[1]*240/224)
            img.draw_rectangle(i1,(0,255,0))
            lcd.display(img)
            #print(i.classid(),i.value())
            xmid = i1[0]+i.w()/2
            ymid = i1[1]+i.h()/2
            pdCount += 1
            totalPos.append((xmid,ymid)) #加入座標
            #print (xmid,ymid)
            if pdCount >= sampleTimes:
                allx = 0.0
                ally = 0.0
                for eachPos in totalPos:
                    allx += eachPos[0]
                    ally += eachPos[1]
                c = (allx / sampleTimes, ally / sampleTimes)
                totalSd = 0.0
                for eachPos in totalPos:
                    totalSd += (pow((eachPos[0] - c[0]), 2) + pow((eachPos[1] - c[1]), 2))
                SD = pow(totalSd / sampleTimes, 0.5)
                for eachPos in totalPos:
                    if pow(pow((eachPos[0]- c[0]), 2)+pow((eachPos[1]- c[1]), 2),0.5) - SD > thresh:
                        totalPos.remove(eachPos)
                allx = 0.0
                ally = 0.0
                for eachPos in totalPos:
                    allx += eachPos[0]
                    ally += eachPos[1]
                c = (allx / sampleTimes, ally / sampleTimes)

                print ('C: ', c)

                #x1 = -(c[0]-224)*21/224
                #y1 = (c[1]-112)*21/224
                x = 11-11/224*c[1]
                y = (c[0] - 112)*18/224
                print("x="+str(x)+","+"y="+str(y))

                gimbal.coor2angle(x,y)
                #if dist(p,c) - SD > thresh:
                #laserPointer(p)
                pdCount = 1
                totalPos = []


    else:
        a = lcd.display(img)
a = kpu.deinit(task)
