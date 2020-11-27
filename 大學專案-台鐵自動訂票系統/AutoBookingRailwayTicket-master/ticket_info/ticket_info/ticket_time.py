class time:
    print("起始時間00:00~23:00 , 截止時間00:00~23:59")
    print("範例：時間為00:00輸入0...01:00輸入1...以此類推 ps:23:59請輸入24")
    def __init__(self):
        return None
    def start_time(self):
        start_time = int(input('輸入起始時間：' ))
        start__time = '//*[@id="getin_start_dtime"]/option[' + str(start_time + 1 ) +']'
        return start__time
    def end_time(self):
        end_time = int(input('輸入截止時間：'))
        end__time = '//*[@id="getin_end_dtime"]/option[' +str(end_time + 1) + ']'
        return end__time