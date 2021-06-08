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


class config:
    batch_size = 8
    zoom_size = 512
    input_size = 448
    start_epoch = 0
    end_epoch = 100
    init_lr = 1e-5
    weight_decay = 1e-5
    momentum = 0.9
    milestones = [20, 40]
    eval_interval = 1
    train_csv = './train.csv'
    valid_csv = './validation.csv'
    data_root = './dataS/low-resolution'

    total_valid = 5200
    num_classes = 130
    checkpoints = './checkpoints-dense'
    logs = './logs-dense'
    # info_file = './logs/densenet.txt'
    results = './results-dense'


class LabelSmoothing(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(LabelSmoothing, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.confidence = 1.0 - epsilon
        self.value = epsilon / (num_classes - 1)

    def execute(self, x, targets):
        batch_size = targets.size(0)
        assert x.size(1) == self.num_classes
        smoothed_labels = jt.full(
            shape=(batch_size, self.num_classes), val=self.value)
        for i in range(batch_size):
            smoothed_labels[i, targets[i]] = self.confidence
        log_prob = jt.nn.log_softmax(x, dim=1)
        out = -(log_prob * smoothed_labels).sum() / batch_size
        return out


def train():
    model = Net()
    model.load_state_dict(jt.load('./trick_model/checkpoints6/30.pkl'))
    criterion = LabelSmoothing(num_classes=config.num_classes)
    # criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.init_lr)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.end_epoch)
    
    train_transform = transform.Compose([
        transform.Resize(512, Image.BILINEAR),
        transform.RandomCrop(448),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transform = transform.Compose([
        transform.Resize(512, Image.BILINEAR),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = TDog(config.train_csv, config.data_root,
                        config.batch_size, transform=train_transform)
    valid_loader = TDog(config.valid_csv, config.data_root,
                        config.batch_size, transform=valid_transform)
    
    for epoch in range(config.end_epoch):
        model.train()
        train_acc1 = 0
        train_loss = 0
        for _, (_, x, y) in enumerate(tqdm(train_loader, desc='Epoch %d Training' % epoch)):
            output = model(x)
            tmp = criterion(output, y)
            optimizer.step(tmp)
            prediction1, _ = output.argmax(dim=1)
            train_acc1 += jt.equal(prediction1, y).sum().float().item()
            train_loss += criterion(output, y).data.tolist()[0]
        for _, (_, x, y) in enumerate(tqdm(valid_loader, desc='Epoch %d Validation' % epoch)):
            output = model(x)
            tmp = criterion(output, y)
            optimizer.step(tmp)
            prediction1, _ = output.argmax(dim=1)
            train_acc1 += jt.equal(prediction1, y).sum().float().item()
            train_loss += nn.cross_entropy_loss(output, y).data.tolist()[0]

        train_acc1 /= 70428 # (65228+5200)
        train_loss /= 70428 # (65228+5200)
        scheduler.step()

        info = '=' * 20 + 'epoch{}'.format(epoch) + '=' * 20 + '\n'
        info += 'train acc {:.4f}\t train loss {:.4f}\n'.format(
            train_acc1, train_loss)
        print(info)
        with open('./logs-dense/{}.txt'.format('train-dense'), 'a+') as f:
            f.write(info)
        jt.save(model.state_dict(), os.path.join(
            './checkpoints-dense', '%d.pkl' % epoch))
        generate(os.path.join('./checkpoints-dense', '%d.pkl' % epoch),
                 os.path.join('./results-dense', '%d.json' % epoch))


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


if __name__ == '__main__':
    jt.flags.use_cuda = 1
    train()
