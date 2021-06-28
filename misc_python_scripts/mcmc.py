import numpy as np
import numba as nb

def check_in_or_out():
   x,y = np.random.random(size=2)
   if x**2 + y**2 <= 1:
      return 1
   else:
      return 0


def estimate_pi(num_total=50000):
	num_in = 0
	for i in range(int(num_total)):
	  num_in += check_in_or_out()
	pi_estimate = 4*num_in/num_total
	return pi_estimate
	
	
print(estimate_pi(50000))
	
@nb.jit(nopython=True)
def estimate_pi_fast(num_total=50000):
	num_in = 0
	for i in range(int(num_total)):
	  x,y = np.random.random(size=2)
	  if x**2 + y**2 <= 1:
	     num_in += 1
	pi_estimate = 4*num_in/num_total
	return pi_estimate


estimate_pi_fast(1e2)
estimates_1e5 = [estimate_pi_fast(1e5) for i in range(100)]
estimates_1e4 = [estimate_pi_fast(1e4) for i in range(100)]
estimates_1e3 = [estimate_pi_fast(1e3) for i in range(100)]

print(np.mean(estimates_1e5), np.std(estimates_1e5))
print(np.mean(estimates_1e4), np.std(estimates_1e4))
print(np.mean(estimates_1e3), np.std(estimates_1e3))


print(np.mean(estimates_1e5)/(2*np.sqrt(1e5)))
print(np.mean(estimates_1e4)/(2*np.sqrt(1e4)))
print(np.mean(estimates_1e3)/(2*np.sqrt(1e3)))


import matplotlib.pyplot as plt
plt.figure()
plt.hist(estimates_1e5, bins=10)
plt.show()
