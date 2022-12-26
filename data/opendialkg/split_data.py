import random
random.seed(123)

with open("opendialkg.csv","r") as f:
    lines = f.readlines()
header = lines[0]
data = lines[1:]

random.shuffle(data)

with open("train.csv","w") as fout:
    fout.write(header)
    fout.writelines(data[:len(data)*70//100])
with open("valid.csv","w") as fout:
    fout.write(header)
    fout.writelines(data[len(data)*70//100:len(data)*85//100])
with open("test.csv","w") as fout:
    fout.write(header)
    fout.writelines(data[len(data)*85//100:])
