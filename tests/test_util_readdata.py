import ares


def test():
    for src in ['mirocha2017', 'bouwens2015', 'finkelstein2015']:
        data = ares.util.read_lit(src)
        
if __name__ == '__main__':
    test()
    
    