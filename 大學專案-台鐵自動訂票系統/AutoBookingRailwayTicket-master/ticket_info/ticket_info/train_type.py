class Type:
    print("全部車種:1 自強號：2 莒光號：3 復興號：4")
    input_type = ''
    def __init__(self):
        return None
    def Type(self):
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