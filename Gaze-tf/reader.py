import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import random
import torch


def dealTheRect(bbox):
    origin = bbox.split(",")

    fx1 = int(origin[0])
    fy1 = int(origin[1])
    fx2 = int(origin[2])
    fy2 = int(origin[3])

    fw = fx2 - fx1
    fh = fy2 - fy1

    lex1 = int(origin[4])
    ley1 = int(origin[5])
    lex2 = int(origin[6])
    ley2 = int(origin[7])

    lew = lex2 - lex1
    leh = ley2 - ley1

    rex1 = int(origin[8])
    rey1 = int(origin[9])
    rex2 = int(origin[10])
    rey2 = int(origin[11])

    rew = (rex2 - rex1)
    reh = (rey2 - rey1)

    rect = [str(fx1 / 1000), str(fy1 / 1000), str(fw / 1000), str(fh / 1000),
            str(lex1 / 1000), str(ley1 / 1000), str(lew / 1000), str(leh / 1000),
            str(rex1 / 1000), str(rey1 / 1000), str(rew / 1000), str(reh / 1000)]

    return rect


def aug_line(line, width, height):
    bbox = np.array(line[2:5])
    bias = round(30 * random.uniform(-1, 1))
    bias = max(np.max(-bbox[0, [0, 2]]), bias)
    bias = max(np.max(-2 * bbox[1:, [0, 2]] + 0.5), bias)

    line[2][0] += int(round(bias))
    line[2][1] += int(round(bias))
    line[2][2] += int(round(bias))
    line[2][3] += int(round(bias))

    line[3][0] += int(round(0.5 * bias))
    line[3][1] += int(round(0.5 * bias))
    line[3][2] += int(round(0.5 * bias))
    line[3][3] += int(round(0.5 * bias))

    line[4][0] += int(round(0.5 * bias))
    line[4][1] += int(round(0.5 * bias))
    line[4][2] += int(round(0.5 * bias))
    line[4][3] += int(round(0.5 * bias))

    line[5][2] = line[2][2] / width
    line[5][3] = line[2][0] / height

    line[5][6] = line[3][2] / width
    line[5][7] = line[3][0] / height

    line[5][10] = line[4][2] / width
    line[5][11] = line[4][0] / height
    return line


def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


class loader(Dataset):
    def __init__(self, path, root, image_size, header=True, ):
        """
        image_size = [224,224,224,224]
        leye reye face origin
        
        """
        self.lines = []
        self.image_size = image_size
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    for k in range(len(line)):
                        line_list = line[k].split(" ")
                        # print(line_list[6])
                        # 边缘也读取
                        
                        if line_list[6] == "Photo":
                            self.lines.append(line[k])
                            # if int(i.split("/")[-1][:5]) > 41100:
                            #     self.lines.append(line[k])
                            #     self.lines.append(line[k])

        else:
            with open(path) as f:
                line = f.readlines()
                if header: self.line.pop(0)
                for j in range(len(line)):
                    line_list = line[j].split(" ")
                    # print(line_list[6])
                    if line_list[6] == "Photo":
                        self.lines.append(line[j])
                        # if int(path.split("/")[-1][:5]) > 41100:
                        #     self.lines.append(line[k])
                        #     self.lines.append(line[k])


        self.root = root

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        kind = line[6]
        name = line[0]
        point = line[5]
        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        grid = line[3]
        full = line[4]
        bbox = line[7]
        # device = line[5]

        label = np.array(point.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        rect = np.array(dealTheRect(bbox)).astype("float")
        rect = torch.from_numpy(rect).type(torch.FloatTensor)
        # print(righteye)
        rimg = cv2.imread(os.path.join(self.root, righteye))
        rimg = cv2.resize(rimg, (self.image_size[0], self.image_size[0])) / 255.0
        rimg = rimg.transpose(2, 0, 1)

        limg = cv2.imread(os.path.join(self.root, lefteye))
        limg = cv2.resize(limg, (self.image_size[1], self.image_size[1])) / 255.0
        limg = limg.transpose(2, 0, 1)

        fimg = cv2.imread(os.path.join(self.root, face))
        fimg = cv2.resize(fimg, (self.image_size[2], self.image_size[2])) / 255.0
        fimg = fimg.transpose(2, 0, 1)

        flimg = cv2.imread(os.path.join(self.root, full))
        flimg = cv2.resize(flimg, (self.image_size[3], self.image_size[3])) / 255.0
        flimg = flimg.transpose(2, 0, 1)

        grid = cv2.imread(os.path.join(self.root, grid), 0)
        grid = np.expand_dims(grid, 0)

        img = {"left": torch.from_numpy(limg).type(torch.FloatTensor),
               "right": torch.from_numpy(rimg).type(torch.FloatTensor),
               "face": torch.from_numpy(fimg).type(torch.FloatTensor),
               "full": torch.from_numpy(flimg).type(torch.FloatTensor),
               # "grid":torch.from_numpy(grid).type(torch.FloatTensor),
               "name": name,
               "rects": rect,
               "label": label,
               "device": "Android"}

        return img


def txtload(labelpath, imagepath, batch_size, image_size, shuffle=True, num_workers=0, header=True):
    dataset = loader(labelpath, imagepath, image_size, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


if __name__ == "__main__":
    image = r"/home/zhuzi/code/data/dataset_output/Image"
    label = r"/home/zhuzi/code/data/dataset_output/Label/train"
    trains = os.listdir(label)
    trains = [os.path.join(label, j) for j in trains]
    # print(trains)
    d = txtload(trains, image, 10,[112,112,224,224])
    print(len(d))

    (load) = d.__iter__()
    print(load)