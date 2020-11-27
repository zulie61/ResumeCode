import numpy as np
import random
class Id:
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