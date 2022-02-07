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

x = np.abs(np.random.normal(0,0.03,size = (1,100)))
q = np.vstack(np.array([-1,-2,-3,-4,-5,-6,-7]))


def test_speed():

    fp0 = timeit.repeat('loopfp0()',globals=globals(), number=1, repeat=200)
    fp1 = timeit.repeat('loopfp1()',globals=globals(), number=1, repeat=200)
    fp2 = timeit.repeat('loopfp2()',globals=globals(), number=1, repeat=200)
    fp3 = timeit.repeat('loopfp3()',globals=globals(), number=1, repeat=200)

    print('########### python ' + sys.version[:6] + '###########')
    print('    regular powers: ' + str(np.round(np.mean(fp0)*1000,2))
          + 'ms ± ' + str(np.round(np.std(fp0)*1000,2))+ 'ms')
    print(' inner float_power: ' + str(np.round(np.mean(fp1)*1000,2))
          + 'ms ± ' + str(np.round(np.std(fp1)*1000,2))+ 'ms')
    print(' outer float_power: ' + str(np.round(np.mean(fp2)*1000,2))
          + 'ms ± ' + str(np.round(np.std(fp2)*1000,2))+ 'ms')
    print('double float_power: ' + str(np.round(np.mean(fp3)*1000,2))
          + 'ms ± ' + str(np.round(np.std(fp3)*1000,2))+ 'ms')
    print('####################################')
