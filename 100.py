import numpy as np
import glob
import sys
import os
import time

dataset = sys.argv[1]
file_num = int(sys.argv[2])
file_dir = sys.argv[3]

res = []
for i in range(file_num):
    ofile = str(i) + '.txt'
    num_txt_path = file_dir
    with open(os.path.join(num_txt_path, ofile), 'r') as f:
        lines = f.readlines()
        cmd = lines[0]
        line = lines[-1]
        line = line.strip().split()[-1]
        res.append(float(line))

res = np.array(res)
print('mean:', np.mean(res))
print('max:', np.max(res))
print('min:', np.min(res))
print('std:', np.std(res))
print("cmd:", cmd)

mean = np.mean(res)
mean = "{:.5f}_".format(mean)
cur_time = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
file_name =  file_dir + "/tune/" + mean + cur_time
f=open(file_name,"a")
f.write(cmd)
f.write(str(res))
f.close()