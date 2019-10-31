# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:54:07 2019

@author: thulfiqar
"""

import numpy as np
import matplotlib.pyplot as plot


x = np.arange(-10, 10, 0.1);

#tanh
y   = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1 )

plot.plot(x, y)

# sigmoid function
y = 1 / (1 + np.exp (-x))

plot.plot(x, y)
