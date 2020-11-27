import numpy as np
import time,random,cv2,os
from ticket_info import ticket_info as info
from selenium import webdriver
from time import sleep
from PIL import Image
from keras.models import load_model
from time import localtime as tlt
start=time.time()
print("--------------------------------------------------")
Id = info.info().Id()
date = info.info().date()
from_station = info.info().from_station()
to_station = info.info().to_station()
Type = info.info().Type()
start_time = info.info().start_time()
end_time = info.info().end_time()
qty = info.info().qty_()

print("-------------------啟動驗證碼解析模型--------------")
#
captcha_height = 128
captcha_width = 128

letter_str = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ_'
letter_amount = len(letter_str)

model_path = 'captcha_model.h5'
dir_path = 'img\\img\\'
#dir_path = 'C:\\workspace\\temp\\test_data\\wrong\\'

print('Loading model : ' + model_path)
model = load_model(model_path)

#model.summary()
print('Load model success\n')


print("-------------------模擬器開啟----------------------")
ntime = 0
failidorseat = 0
failcaptcha = 0
unknow = 0
success = 0

def railwaykill():
    while True:
        global success
        global ntime
        global failidorseat
        global failcaptcha
        global Id
        global date
        global from_station
        global to_station
        global Type
        global start_time
        global end_time
        global qty
        #開啟台鐵網
        print("---------------開始訂票-------------------")
        browser= webdriver.Chrome()
        browser.get('http://railway.hinet.net/Foreign/TW/etkind1.html')
        #輸入身分證字號
        browser.find_element_by_name('person_id').send_keys(Id)
        #sleep(1)
        #點擊乘車日期
        browser.find_element_by_xpath(date).click()
        #sleep(1)
        #點擊起訖站
        browser.find_element_by_xpath(from_station).click()
        browser.find_element_by_xpath(to_station).click()
        #sleep(1)
        #點擊車種
        browser.find_element_by_xpath(Type).click()
        #sleep(1)
        #點集起始時間與票數
        browser.find_element_by_xpath(start_time).click()
        browser.find_element_by_xpath(end_time).click()
        browser.find_element_by_xpath(qty).click()
        sleep(1)
        #開始訂票
        browser.find_element_by_class_name('btn-primary').submit()
        #擷取整個網頁畫面
        browser.get_screenshot_as_file("img\\img.jpg")
        
        print("----------------驗證碼解析----------------")
        #尋找驗證碼的位置
        picture = browser.find_element_by_id('idRandomPic')
        picture.location #{'x': 46, 'y': 455}
        picture.size #{'height': 70, 'width': 210}
        left  = picture.location['x']
        right = picture.location['x'] + picture.size['width']
        top   = picture.location['y']
        bottom= picture.location['y'] + picture.size['height']
        #46 256 455 525
        #把驗證碼從網頁畫面切割下來
        img = Image.open("img\\img.jpg")
        img = img.crop((left+6,top+5,right-4,bottom-5))
        #img.show()
        img = img.convert('RGB')
        img.save("img\\img\\img.jpg",'jpeg')
        data = np.zeros([1, captcha_height, captcha_width, 3]).astype('float32')
        for file in os.listdir(dir_path) :
        
            src = cv2.imread(dir_path + file)    
            img = np.zeros((captcha_height, captcha_width, 3), np.uint8)
            img[0:60, 14:114] = src[0:60, 0:100]
            img[68:128, 14:114] = src[0:60, 100:200]
            
            data[0] = img
            data = data / 255
            
            predict = model.predict(data)
            captcha = ''
            if predict[0][0].argmax(axis=-1) == 1 :
                for i in range(1, 6) :
                    index = predict[i][0].argmax(axis=-1)
                    captcha = captcha + letter_str[index]
            elif predict[0][0].argmax(axis=-1) == 0 :
                for i in range(6, 12) :
                    index = predict[i][0].argmax(axis=-1)
                    captcha = captcha + letter_str[index]
            
            print('result : ' + captcha)
            
        #    cv2.imshow('src', src)
        #    cv2.waitKey(0)
        browser.find_element_by_name('randInput').send_keys(captcha)
        browser.find_element_by_class_name('btn-primary').submit()
        sleep(1)
        ntime+=1
        wait=random.randint(6,12)
        check = browser.find_element_by_class_name('page-header')
        check = check.size
        failcheck = browser.find_element_by_xpath("/html/body/div/div[3]/div")
        failcheck = failcheck.size
        def show_localtime():
            print('執行程式時間為：%d年%d月%d日%d點%d分%d秒' %(tlt()[0],tlt()[1],tlt()[2],tlt()[3],tlt()[4],tlt()[5]))
        print("---------------訂票結果-------------------")
        if check["height"] == 140 :
            browser.get_screenshot_as_file("success\\img.jpg")
            success +=1
            print("訂票成功!")
            show_localtime()
            break     
        else:
            if failcheck['height'] == 280  :
                print("訂票失敗")
                print("亂數驗證失敗")
                print(str(wait) + "秒後將重新進行訂票")
                failcaptcha+=1
                sleep(1)
                browser.close()
                sleep(wait)
            else :
                print("訂票失敗")
                print("班次額滿或身分證錯誤")
                print(str(wait) + "秒後將重新進行訂票")
                failidorseat+=1
                sleep(1)
                browser.close()
                sleep(wait)
        print("總訂票次數為%d次" % (ntime))
        print("訂票成功次數:" + str(success))
        print("班次額滿或身分證錯誤為%d次" % (failidorseat))
        print("驗證碼錯誤為%d次" % (failcaptcha))
        show_localtime()
try:
    railwaykill()
except:
    unknow += 1
    print("訂票失敗")
    print("不可預期的失誤%d次" % (unknow))
    railwaykill()
finally:
    railwaykill()
end=time.time()
elapsed =end - start 
print("執行時間:%.1f秒" % (elapsed))
