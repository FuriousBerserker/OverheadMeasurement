#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

thread_block = np.zeros(0, dtype=int)
original = np.zeros(0, dtype=float)
shadow_memory = np.zeros(0, dtype=float)

if (not sys.argv[1]):
    print("No input file")
    exit()
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        name, value = line.split()
        offset = i % 3;
        if (offset == 0):
            thread_block = np.append(thread_block, int(value))
        elif (offset == 1):
            original = np.append(original, float(value))
        else:
            shadow_memory = np.append(shadow_memory, float(value))
#print(thread_block)
#print(original)
#print(shadow_memory)

ydata = shadow_memory / original

fig, ax = plt.subplots()
ax.plot(thread_block, ydata, color = '#89DBCB', marker = '*')
ax.set(xlabel = 'Number of Thread Block', ylabel = 'Slowdown', title = 'Overhead with Conflict Benchmark')
ax.grid()
fig.savefig('conflict-overhead.pdf', format = 'pdf')
plt.show()
