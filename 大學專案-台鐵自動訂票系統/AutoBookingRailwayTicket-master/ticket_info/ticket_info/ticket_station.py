from bs4 import BeautifulSoup
import requests
class station:
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
    def __init__(self):
        return None
    def from_station(self):        
        from_ = int(input("輸入起站代碼: "))
        from_xpath = '//*[@id="from_station"]/option[' + str(from_+1)+ ']'
        return from_xpath
    def to_station(self):
        to_ = int(input("輸入到站代碼: "))
        to_xpath = '//*[@id="to_station"]/option[' + str(to_+1) +']'
        return to_xpath