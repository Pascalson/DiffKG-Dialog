import pandas as pd
import json

df = pd.read_csv("test.csv",converters={'Messages':json.loads})

for dialogue in df["Messages"]:
    for turn in dialogue:
        if turn["type"] == "chat":
            print("{}: {}".format(turn["sender"],turn["message"]))
            if turn["sender"] == "assistant":
                Rtype = input(\
                    "Enter Reasoning type:\n"
                    "{0:'no-reasoning-required',1:'inform',2:'selection',3:'extraction',4:'true/false'} or other names\n")
                turn["reasoning-type"] = Rtype
                print("have recorded the Rtype: {}".format(Rtype))
#TODO: save the json file after every dialogue
