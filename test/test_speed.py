import sys
sys.path.append("../")

import numpy as np
import timeit

# tests powers
def loopfp0():
    for i in range(1000):
        np.mean(x**(q/2), axis=1)**(1/q.T)
    return

def loopfp1():
    for i in range(1000):
        np.mean(np.float_power(x, q/2), axis=1)**(1/q.T)
    return

def loopfp2():
    for i in range(1000):
        np.float_power(np.mean(x**(q/2), axis=1),1/q.T)
    return

def loopfp3():
    for i in range(1000):
        np.float_power(np.mean(np.float_power(x, q/2), axis=1),1/q.T)
    return

# tests powers (longer sequence)
def loopfp0_l():
    for i in range(1000):
        np.mean(x_l**(q/2), axis=1)**(1/q.T)
    return

def loopfp1_l():
    for i in range(1000):
        np.mean(np.float_power(x_l, q/2), axis=1)**(1/q.T)
    return

def loopfp2_l():
    for i in range(1000):
        np.float_power(np.mean(x_l**(q/2), axis=1),1/q.T)
    return

def loopfp3_l():
    for i in range(1000):
        np.float_power(np.mean(np.float_power(x_l, q/2), axis=1),1/q.T)
    return

x = np.abs(np.random.normal(0,0.03,size = (1,100)))
x_l = np.abs(np.random.normal(0,0.03,size = (1,1000)))
q = np.vstack(np.array([-1,-2,-3,-4,-5,-6,-7]))


def test_speed():

    fp0 = timeit.repeat('loopfp0()',globals=globals(), number=1, repeat=50)
    fp1 = timeit.repeat('loopfp1()',globals=globals(), number=1, repeat=50)
    fp2 = timeit.repeat('loopfp2()',globals=globals(), number=1, repeat=50)
    fp3 = timeit.repeat('loopfp3()',globals=globals(), number=1, repeat=50)

    fp0_l = timeit.repeat('loopfp0_l()',globals=globals(), number=1, repeat=25)
    fp1_l = timeit.repeat('loopfp1_l()',globals=globals(), number=1, repeat=25)
    fp2_l = timeit.repeat('loopfp2_l()',globals=globals(), number=1, repeat=25)
    fp3_l = timeit.repeat('loopfp3_l()',globals=globals(), number=1, repeat=25)

    print('                    python: ' + sys.version[:6]
          +    '  numpy: ' + np.__version__)
    print('(time per iteration)       N = 100                N = 1000')
    print('    regular powers: ' + '{:.2f}'.format(np.mean(fp0)*10000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp0)*10000) + 'µs     '
          + '{:.2f}'.format(np.mean(fp0_l)*1000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp0_l)*1000) + 'µs')
    print(' inner float_power: ' + '{:.2f}'.format(np.mean(fp1)*10000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp1)*10000) + 'µs     '
          + '{:.2f}'.format(np.mean(fp1_l)*1000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp1_l)*1000) + 'µs')
    print(' outer float_power: ' + '{:.2f}'.format(np.mean(fp2)*10000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp2)*10000) + 'µs     '
          + '{:.2f}'.format(np.mean(fp2_l)*1000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp2_l)*1000) + 'µs')
    print('double float_power: ' + '{:.2f}'.format(np.mean(fp3)*10000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp3)*10000) + 'µs     '
          + '{:.2f}'.format(np.mean(fp3_l)*1000)
          + 'µs ± ' + '{:.2f}'.format(np.std(fp3_l)*1000) + 'µs')
