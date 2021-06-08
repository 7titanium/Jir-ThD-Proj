import json
import sys

ans = {}
load_dict = {}
file_path = sys.argv[1]

with open("./lables.json",'r', encoding='UTF-8') as f:
    ans = json.load(f)
with open(file_path,'r', encoding='UTF-8') as f:
    load_dict = json.load(f)
total = len(ans)
top1 = 0
top5 = 0
for k,v in ans.items():
    r = load_dict[k]
    if r[0] == v:
        top1 += 1
        top5 += 1
    elif v in r:
        top5 += 1

top1 = top1 / total
top5 = top5 / total
print("top1: ",top1 ,"\ntop5: ",top5)