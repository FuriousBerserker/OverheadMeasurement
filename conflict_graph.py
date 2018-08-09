#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

thread_block = np.zeros(0, dtype=int)
original = np.zeros(0, dtype=float)
shadow_memory_atomic = np.zeros(0, dtype=float)
shadow_memory_volatile = np.zeros(0, dtype=float)

group_size = 4
all_data=[original, shadow_memory_atomic, shadow_memory_volatile]
colors = ['#66CC00', '#00CCCC']
markers = ['*', 'o']
legends = ['shadow_memory_atomic', 'shadow_memory_volatile']

if (not sys.argv[1]):
    print("No input file")
    exit()
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        name, value = line.split()
        offset = i % group_size;
        if (offset == 0):
            thread_block = np.append(thread_block, int(value))
        else:
            all_data[offset - 1] = np.append(all_data[offset - 1], float(value))

for i in range(1, len(all_data)):
    all_data[i] /= all_data[0]

#print(thread_block)
#print(all_data)
lines = []
fig, ax = plt.subplots()
max_yaxis = 0;
for i in range(0, len(legends)):
    line = ax.plot(thread_block, all_data[i + 1], color = colors[i], marker = markers[i], label=legends[i])
    lines.append(line)
    max_yaxis = max(max_yaxis, all_data[i + 1].max())
ax.set(xlabel = 'Number of Thread Block', ylabel = 'Slowdown', title = 'Overhead with Conflict Benchmark')
ax.set_xlim(0, thread_block[len(thread_block) - 1] + 5)
ax.set_ylim(0, int(max_yaxis * 1.1))
ax.grid()
ax.legend()
fig.savefig('conflict-overhead.pdf', format = 'pdf')
#plt.show()
