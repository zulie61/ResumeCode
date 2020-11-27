import datetime
import time
from bs4 import BeautifulSoup
import requests
import numpy as np
import random

class info:

    def __init__(self):
        return None
    def Id(self):
        #計算身分證英文所對應的數字
        countyone = np.array([1,1,1,1,1,1,1,1,3,1,1,2,2,2,3,2,2,2,2,2,2,2,3,3,3,3])
        countytwo = np.array([0,1,2,3,4,5,6,7,4,8,9,0,1,2,5,3,4,5,6,7,8,9,2,0,1,3])
        countyadd = countyone + 9*countytwo 
        #隨機生成身分證數字 除了最後一位
        def rm():
            rm = random.randint(0,9)
            return rm
        idnum=[countyadd[random.randint(0,25)],random.randint(1,2)*8,rm()*7,rm()*6,rm()*5,rm()*4,rm()*3,rm()*2,rm(), 0]
        #生成最後一位數字
        num = 0 
        for i in range(0,9):
            num += idnum[i]
        idnum[9]=10-num%10
        #將英文所對應的數字轉換回去英文
        countyeng=['A','B','C','D','E','F','G','H','I','J','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        county={}
        for j in range(0,26):
            county[countyeng[j]] = countyadd[j] 
        ##change key and value
        county_2= {v:k for k,v in county.items()} 
        idnum[0]=county_2[idnum[0]]
        #把上面rm乘的數字除掉 #idnum[1]=idnum[1]//8 #idnum[2]=idnum[2]//7
        for i in range(1,8):
            idnum[i]=idnum[i]//(9-i)
        #上觸身分證資料為串列 #串列轉換為字串
        identify=''
        for i in range(len(idnum)):
            identify += str(idnum[i])
        return identify
    def date(self):
        year,month,day = map(int,input("輸入乘車日期(年 月 日 以空格區分):").split())
        def ticket_time(year, month, day): 
          date = datetime.date(year, month, day)
          return date.strftime('%j')
        today_time = time.localtime()[7]
        until = int(ticket_time(year,month,day)) - int(today_time)
        time_xpath ='//*[@id="getin_date"]/option[' + str(until+1) +"]"
        return time_xpath
    def from_station(self):
        url =   "http://railway.hinet.net/Foreign/TW/etno1.html"
        html = requests.get(url)
        html.encoding='utf-8'
        sp = BeautifulSoup(html.text,"html.parser")
            
            
        data1=sp.select("#from_station")
        data2=str(data1[0])
        fomart = ' abcdefghijklmnopqrstuvwxyz0123456789<>=/_,:"' 
        for _ in data2:
            if _ in fomart :
                data2 = data2.replace(_, '' )
        data2 = data2.split('-')
        data2.remove(data2[0])
        data2.remove(data2[0])
        for i in range(len(data2)):
            data2[i]=data2[i].strip()
            
            
        place={}
        for i in range(len(data2)):
            place[data2[i]] = i
        for i in range(len(data2)):
            print(data2[i]+':',i,'  ',end='')
            if i%6 ==0 :
                print('\n')
        from_ = int(input("輸入起站代碼: "))
        from_xpath = '//*[@id="from_station"]/option[' + str(from_+1)+ ']'
        return from_xpath
    def to_station(self):
        to_ = int(input("輸入到站代碼: "))
        to_xpath = '//*[@id="to_station"]/option[' + str(to_+1) +']'
        return to_xpath
    def start_time(self):
        print("起始時間00:00~23:00 , 截止時間00:00~23:59")
        print("範例：時間為00:00輸入0...01:00輸入1...以此類推 ps:23:59請輸入24")
        start_time = int(input('輸入起始時間：' ))
        start__time = '//*[@id="getin_start_dtime"]/option[' + str(start_time + 1 ) +']'
        return start__time
    def end_time(self):
        end_time = int(input('輸入截止時間：'))
        end__time = '//*[@id="getin_end_dtime"]/option[' +str(end_time + 1) + ']'
        return end__time
    def Type(self):
        print("全部車種:1 自強號：2 莒光號：3 復興號：4")
        choose_type = int(input("選擇車種(輸入對應的數字)："))
        if choose_type == 1 :
            input_type = '//*[@id="train_type"]/option[1]'
        elif choose_type == 2 :
            input_type = '//*[@id="train_type"]/option[2]'
        elif choose_type == 3 :
            input_type = '//*[@id="train_type"]/option[3]'
        elif choose_type == 4 :
            input_type = '//*[@id="train_type"]/option[4]'
        else :
            input_type = "Error 無對應車種"
        return input_type
    def qty_(self):
        qty_str = int(input("訂票張數(1~6)："))
        qty__str = '//*[@id="order_qty_str"]/option['+str(qty_str) + ']'
        return qty__str
