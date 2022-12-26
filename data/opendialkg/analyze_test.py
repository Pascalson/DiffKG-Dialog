import pandas as pd
import json
import pdb

df = pd.read_csv("test.csv",converters={'Messages':json.loads})

kgwalk_times = []
potential_types = []
for dialogue in df["Messages"]:
    kgwalk_times_this_dialogue = 0
    potential_type = []
    for turn in dialogue:
        if turn["type"] == "chat":
            if "music" in turn["message"].lower():
                potential_type.append("music")
            if "sport" in turn["message"].lower():
                potential_type.append("sport")
            if "book" in turn["message"].lower():
                potential_type.append("book")
            if "movie" in turn["message"].lower():
                potential_type.append("movie")
        if turn["type"] == "action":
            kgwalk_times_this_dialogue += 1
    potential_types.append(list(set(potential_type)))
    kgwalk_times.append(kgwalk_times_this_dialogue)

print("KGWalk Times")
kgwalk_num_dict = {}
for k in kgwalk_times:
    if k not in kgwalk_num_dict:
        kgwalk_num_dict[k] = 0
    kgwalk_num_dict[k] += 1
for key, value in sorted(kgwalk_num_dict.items(), key=lambda x: x[0]):
    print("{} : {}".format(key, value))
print("MIN:{}".format(min(kgwalk_times)))
print("MAX:{}".format(max(kgwalk_times)))
print("AVG:{}".format(sum(kgwalk_times)/len(kgwalk_times)))


types_dict = {"book":0,"movie":0,"music":0,"sport":0}
for t in potential_types:
    for key in t:
        types_dict[key] += 1

types_num_dict = {}
for t in potential_types:
    if len(t) not in types_num_dict:
        types_num_dict[len(t)] = 0
    types_num_dict[len(t)] += 1


print("Potential Types")
print(types_dict)
for key, value in sorted(types_num_dict.items(), key=lambda x: x[0]):
    print("{} : {}".format(key, value))

print("MAX NUM TYPES: {}".format(max(len(t) for t in potential_types)))
print("ZERO TYPES NUM: {}".format(sum(len(t) == 0 for t in potential_types)))

pdb.set_trace()
