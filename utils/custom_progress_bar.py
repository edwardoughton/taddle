
class CustomProgressBar:
    """
    Made a custom progress "bar" because tqdm was slowing down my forward pass speed by a 2x factor...
    Really just a rudimentary carriage return printer :)
    """
    def __init__(self, total):
        self.total = total
        self.cur = 0
        
    def update(self, amount):
        self.cur += amount
        print(f'\t {round(self.cur*100/self.total, 2)}%', end='\r')
        