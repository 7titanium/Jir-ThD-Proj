from posix import listdir
import jittor as jt 
import jittor.nn as nn 
from dataset import TsinghuaDog
from jittor import transform
from jittor.optim import Adam, SGD
from tqdm import tqdm
import numpy as np
from model import Net
import argparse 
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os
from dogDict import dogDict
import json




jt.flags.use_cuda=1



transform_test = transform.Compose([
        transform.Resize((512, 512)),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

model_path = './best_model.pkl'


new_model = Net(num_classes=130)
new_model.load_parameters(jt.load(model_path))


pre = {}

# test_path = '/home/jerry/DATA/ThuDogs/low-resolution/1043-n000001-Shiba_Dog/'
# test_path = '/home/jerry/DATA/ThuDogs/low-resolution/203-n000022-English_setter/'
test_path = './MY_R/MY_TEST/'


new_model.eval()


pre = {}

images = []
pred = []
count = 0

for img in tqdm(os.listdir(test_path)):
    image0 = Image.open(test_path+img).convert('RGB')
    image0 = transform_test(image0)

    img0 = jt.array(np.asarray(image0)[np.newaxis, :])

    output = new_model(img0)
    output = output.detach().numpy()

    top5 = np.argsort(-output,axis=1)[0][:5]

    dogN = dogDict[str(top5[0]+1)]

    images.append(img0[0])
    pred.append(dogN)

    pre.update({
        img:list(top5)
    })


    
for k,v in pre.items():
    pre[k] = [ int(v[i])+1 for i in range(len(v))]
    
item = json.dumps(pre)
with open('./result.json','w') as f:
    f.write(item)