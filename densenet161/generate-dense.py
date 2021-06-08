import json
import jittor as jt
import jittor.nn as nn
from jittor import transform
from jittor.optim import Adam, SGD
from jittor.dataset import Dataset
import jittor.models as models
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from PIL import Image, ImageFilter, ImageDraw
import random

class TDog(Dataset):
    def __init__(self, csv, root_dir, batch_size, shuffle=True, transform=None, flag=False):
        super().__init__()
        self.root_dir = root_dir
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.generate = flag
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.df),
            shuffle=shuffle
        )

    def __getitem__(self, idx):
        name = self.df.iloc[idx, 1]
        img = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
        if not self.generate:
            if random.random() < 0.5:  # 滤镜
                img = img.filter(ImageFilter.FIND_EDGES)
            elif random.random() < 0.5:  # 模糊
                img = img.filter(ImageFilter.GaussianBlur(10))
            elif random.random() < 0.5:  # 线条
                img_d = ImageDraw.Draw(img)
                x_len, y_len = img.size
                color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))
                for x in range(0, x_len, 10):
                    img_d.line(((x, 0), (x, y_len)), color)
                for y in range(0, y_len, 10):
                    img_d.line(((0, y), (x_len, y)), color)
            else:
                pass
        img = self.transform(img)
        img = np.asarray(img)
        label = self.df.iloc[idx, 2]
        return name, img, label - 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        densenet = models.densenet161(pretrained=False)
        densenet.load_state_dict(
            jt.load('./densenet161.pkl'))
        self.backbone2 = densenet.features
        print(densenet.classifier.in_features) # 2208
        self.classifier2 = nn.Linear(densenet.classifier.in_features, 130)

    def execute(self, x):
        out = self.backbone2(x)
        out = nn.relu(out)
        out = jt.pool.pool(out, kernel_size=14, op="mean", stride=1)
        # print(out.shape) # [8,2208,1,1,]
        out = out.reshape([out.shape[0], -1])
        # print(out.shape) # [8,2208,]
        out = self.classifier2(out)
        return out


def generate(model_path, result_path):
    model = Net()
    model.load_state_dict(jt.load(model_path))
    model.eval()
    test_transform = transform.Compose([
        transform.Resize(512, Image.BILINEAR),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_loader = TDog(config.csv, config.root, 8,
                       transform=test_transform, flag=True)
    result = dict()
    for name, img, _ in tqdm(test_loader, desc='Generating...'):
        output = model(img)
        _, prediction5 = output.topk(5, 1, True, True)
        for i in range(len(name)):
            p5 = prediction5[i].data.tolist()
            p5 = [j + 1 for j in p5]
            result[name[i]] = p5
    json.dump(result, open(result_path, 'w'))

