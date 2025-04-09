"""
 LegendreShiftPoly.py by @author: kkuriyan, 01-10-2018 translated from
 LegendreShiftPoly.m by Peter Roche, 12-08-2004
 Based on recurrence relation
 (n + 1)Pn+1 (x) - (1 + 2 n)(2 x - 1)Pn (x) + n Pn-1 (x) = 0

 Given nonnegative integer n, compute the 
 Shifted Legendre polynomial P_n.
 Return the result as a vector whose mth
 element is the coefficient of x^(n+1-m).
 polyval(LegendreShiftPoly(n),x) evaluates P_n(x).

"""

import numpy as np
import math

def LegendreShiftPoly(n):

	if n == 0:
		pk = np.array([1])
	elif n == 1:
		pk = np.array([2, -1])
		pk = math.sqrt(3) * pk
	else:

		pkm2 = np.zeros((n+1,))
		pkm2[n] = 1
		pkm1 = np.zeros((n+1,))
		pkm1[n] = -1
		pkm1[n-1] = 2

		for k in range(2,n+1):

			pk = np.zeros((n+1,))

			for e in range(n-k,n):
				pk[e] = (4*k-2)*pkm1[e+1] + (1-2*k)*pkm1[e] + (1-k)*pkm2[e]

			pk[n] = (1-2*k)*pkm1[n] + (1-k)*pkm2[n]
			pk = pk/k

			if k<n:
				pkm2 = pkm1
				pkm1 = pk

		pk = math.sqrt(2*n+1)*pk

	return pk


if __name__ == '__main__' :
	n = sys.argv[1]
	yout = LegendreShiftPoly(n)
	print yout
