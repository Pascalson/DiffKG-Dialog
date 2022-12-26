import json
import os

triples_num = []
for i in range(303):
    path = "D{}_ids.json".format(i)
    if os.path.exists(path):
        with open(path,"r") as fin:
            kg = json.load(fin)
            triples_num.append(len(kg))
print(triples_num)
print(min(triples_num))
print(max(triples_num))
print(sum(triples_num)/len(triples_num))
