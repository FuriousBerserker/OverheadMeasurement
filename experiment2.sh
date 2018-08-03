#!/bin/sh

#set -x
set -e

block_num_start=1
block_num_delta=5
block_num_limit=100

now=$(date +'%Y%m%d-%H%M%S')
output_file="conflict-${now}.log"
for block_num in $(seq ${block_num_start} ${block_num_delta} ${block_num_limit});
do
        echo "Thread block num = ${block_num}"
        echo "thread_block ${block_num}" >> ${output_file}
        echo "original:"
        MACRO_FOR_CONFLICT="-DBLOCK_NUM=${block_num}" make -B conflict_cu.exe >/dev/null 2>&1
        ./conflict_cu.exe | grep "Min:" | awk '{print "original " $6}' | tee -a ${output_file}
        echo "shadow memory:"
        MACRO_FOR_CONFLICT="-DBLOCK_NUM=${block_num} -DSHADOW_MEMORY=1" make -B conflict_cu.exe >/dev/null 2>&1
        ./conflict_cu.exe | grep "Min:" | awk '{print "shadow_memory " $6}' | tee -a ${output_file}
done
./conflict_graph.py ${output_file}
