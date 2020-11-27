class qty:
    def __init__(self):
        return None
    def qty_(self):
        qty_str = int(input("訂票張數(1~6)："))
        qty__str = '//*[@id="order_qty_str"]/option['+str(qty_str) + ']'
        return qty__str