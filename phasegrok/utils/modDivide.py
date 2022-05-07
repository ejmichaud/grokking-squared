# Python3 program to do modular division
import math
from numpy import vectorize
import numpy as np
# Function to find modulo inverse of b. It returns
# -1 when inverse doesn't
# modInverse works for prime m


def _modInverse(b, m):
    g = math.gcd(b, m)
    if (g != 1):
        # print("Inverse doesn't exist")
        return -1
    else:
        # If b and m are relatively prime,
        # then modulo inverse is b^(m-2) mode m
        return (b ** (m - 2)) % m


# Function to compute a/b under modulo m
def _modDivide(a, b, m):
    a = a % m
    inv = _modInverse(b, m)
    result = (inv * a) % m if inv != -1 else -1
    return result


modDivide = vectorize(_modDivide)
