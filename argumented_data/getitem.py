# using in TDog(nn.Dataset)

# normal

def __getitem__(self, idx):
    name = self.df.iloc[idx, 1]
    img = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
    if self.transform:
        img = self.transform(img)
    img = np.asarray(img)
    label = self.df.iloc[idx, 2]
    return name, img, label - 1

# filter
def __getitem__(self, idx):
    name = self.df.iloc[idx, 1]
    img = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
    if self.blur:
        img = img.filter(ImageFilter.FIND_EDGES)
    if self.transform:
        img = self.transform(img)
    img = np.asarray(img)
    label = self.df.iloc[idx, 2]
    return name, img, label - 1

# fuzzy
    def __getitem__(self, idx):
        name = self.df.iloc[idx, 1]
        img = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
        if self.blur:
            img = img.filter(ImageFilter.GaussianBlur(10))
        if self.transform:
            img = self.transform(img)
        img = np.asarray(img)
        label = self.df.iloc[idx, 2]
        return name, img, label - 1

# BB head
    def __getitem__(self, idx):
        name = self.df.iloc[idx, 1]
        img = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
        if self.blur:
            path = '/usr/src/jittor/triple-z/JTDog/data/Low-Annotations' + name[1:] + '.xml'
            dom_tree = xml.dom.minidom.parse(path)
            annotation = dom_tree.documentElement
            obj = annotation.getElementsByTagName('object')[0]
            box = obj.getElementsByTagName('headbndbox')[0]
            xmin = int(box.getElementsByTagName('xmin')[0].childNodes[0].data)
            ymin = int(box.getElementsByTagName('ymin')[0].childNodes[0].data)
            xmax = int(box.getElementsByTagName('xmax')[0].childNodes[0].data)
            ymax = int(box.getElementsByTagName('ymax')[0].childNodes[0].data)
            img = img.crop((xmin, ymin, xmax, ymax))
        if self.transform:
            img = self.transform(img)
        img = np.asarray(img)
        label = self.df.iloc[idx, 2]
        return name, img, label - 1

# line
    def __getitem__(self, idx):
        name = self.df.iloc[idx, 1]
        img = Image.open(os.path.join(self.root_dir, name)).convert('RGB')
        if self.blur:
            img_d = ImageDraw.Draw(img)
            x_len, y_len = img.size
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for x in range(0, x_len, 10):
                img_d.line(((x, 0), (x, y_len)), color)
            for y in range(0, y_len, 10):
                img_d.line(((0, y), (x_len, y)), color)
        if self.transform:
            img = self.transform(img)
        img = np.asarray(img)
        label = self.df.iloc[idx, 2]
        return name, img, label - 1