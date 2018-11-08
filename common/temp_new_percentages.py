
# import subprocess
#
# cmd = ["which","nvidia-smi"]
# p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
# res = p.stdout.readlines()
# print(res)
# if len(res) == 0:
#     print(False)
# print(True)

import pandas


def p2f(x):
    # adding this right now bc I saw an error, will think about better way to fix.
    try:
        return float(x.strip('%'))
    except Exception as e:
        print(e)
        return 0.00

# df = pandas.read_csv('/Users/dcarron/Documents/gpu_out.txt', converters={' utilization.memory [%]':p2f,  ' utilization.gpu [%]':p2f})
# print(df.columns)

import os, sys

# we are going to pass in the filename as first param
fn = sys.argv[1]

# fn = '/Users/dcarron/Documents/gpu_out_end.txt'
with open(fn) as file:
    lines = file.readlines()


with open(fn+'stripped','w') as file:
    file.writelines([item for item in lines[:-10]])



df = pandas.read_csv(fn+'stripped')
print(df.shape)
df = df[df['index'].isin(['0','1','2','3','4','5','6','7','8','9','10','11'])]
print(df.shape)
df[' utilization.memory [%]'] = df[' utilization.memory [%]'].apply(p2f)
df[' utilization.gpu [%]'] = df[' utilization.gpu [%]'].apply(p2f)
m = df.loc[df[' utilization.memory [%]'] > 0].groupby(['index'])[' utilization.gpu [%]', ' utilization.memory [%]'].mean()
print(m)

with open(fn+'stripped', 'a+') as file:
    file.write(str(m))

a = 10

# with open('/Users/dcarron/Documents/gpu_out.txt', 'a+') as file:
#     file.write(str(m))
