import datetime
import time
class date:
    def __init__(self):
        return None
    def date(self):
        year,month,day = map(int,input("輸入乘車日期(年 月 日 以空格區分):").split())
        def ticket_time(year, month, day): 
          date = datetime.date(year, month, day)
          return date.strftime('%j')
        today_time = time.localtime()[7]
        until = int(ticket_time(year,month,day)) - int(today_time)
        time_xpath ='//*[@id="getin_date"]/option[' + str(until+1) +"]"
        return time_xpath