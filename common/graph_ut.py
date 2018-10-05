import matplotlib
import matplotlib.pyplot as plt

path = '/Users/dcarron/Documents/cpu_out_wash.txt'

with open(path) as file:
    lines = file.readlines()

print(lines)

tmp = []
for x in lines:
    split = x.split('|')
    print(float(split[1]))
    tmp.append(float(split[1]))

plt.plot(tmp)
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100))
plt.show()