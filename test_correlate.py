import numpy as np
from matplotlib import pyplot as plt

def main():
    data = [5,4,3,2,1,0,1,2,3,4,5,6]
    part = [5,4,3,2]
    res = np.correlate(data + data, part)
    idx = np.argmax(res)
    plt.plot(data + data)
    plt.plot(data)
    plt.plot(part)
    plt.plot(res)
    print(idx)
    print(res)
    print([1,2,3] + [1,2,3])
    plt.show()




main()