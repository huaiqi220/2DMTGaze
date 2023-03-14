
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
import math
import importlib
from model import load_model

def dis(p1, p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

if __name__ == "__main__":
    device_ids = [1,2,3]
    config = yaml.load(open("config/gazetr_config.yaml"), Loader=yaml.FullLoader)
    readername = config["reader"]
    configt = config["train"]
    model_config = configt['model']
    config = config["test"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["load"]["model_name"]
    load_path = os.path.join(config["load"]["load_path"])
    modelname = configt["model"]["model_name"]
    dataloader = importlib.import_module("reader-dir." + readername)
    now_train = str(modelname) + "/bs_"  + str(configt["params"]["batch_size"])  + "_ep_" + str(str(configt["params"]["epoch"])) + "_lr_" + str(str(configt["params"]["lr"])) \
         +"_1"


    device = torch.device("cuda:"+ str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    save_name="evaluation"

    print(f"Test Set: tests")

    tests = os.listdir(labelpath)
    tests = [os.path.join(labelpath, j) for j in tests]

    save_path = os.path.join(config["load"]["load_path"],"checkpoint", now_train)

    if not os.path.exists(os.path.join(load_path, save_name, modelname)):
        os.makedirs(os.path.join(load_path, save_name, modelname))

    print("Read data")

    # print(configt["model"]["image_size"])
    dataset = dataloader.txtload(tests, imagepath, 32, image_size=configt["params"]["image_size"], shuffle=False, num_workers=64, header=True)

    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]
    epoch_log = open(os.path.join(load_path, f"{save_name}/epoch.log"), 'a')
    for save_iter in range(begin, end+step, step):
        print("Model building")
        net = load_model.loadTheModel(model_config)
        net = nn.DataParallel(net,device_ids)
        state_dict = torch.load(os.path.join(save_path, f"Iter_{save_iter}_{modelname}.pt"))
        net.load_state_dict(state_dict)
        net=net.module
        net.to(device)
        net.eval()
        print(f"Test {save_iter}")
        length = len(dataset)
        total = 0
        count = 0
        loss_fn = torch.nn.MSELoss()
        SE_log = open('./SE.log', 'w')
        with torch.no_grad():
            with open(os.path.join(load_path, f"{save_name}/{save_iter}.log"), 'w') as outfile:
                outfile.write("subjcet,name,x,y,labelx,labely,error\n")
                for j, data in enumerate(dataset):
                    batch = {}
                    batch["faceImg"] = data["face"].to(device)
                    batch["leftEyeImg"] = data["left"].to(device)
                    batch['rightEyeImg'] = data['right'].to(device)
                    batch['rects'] = data['rects'].to(device)
                    batch['origin'] = data['full'].to(device)
                    labels = data["label"].to(device)
                    gazes = net(batch, device)
                    
                    names = data["name"]
                    print(f'\r[Batch : {j}]', end='')
                    #print(f'gazes: {gazes.shape}')
                    for k, gaze in enumerate(gazes):
                        #print(f'gaze: {gaze}')
                        gaze = gaze.cpu().detach()
                        count += 1
                        acc = dis(gaze, labels[k])
                        total += acc
                        gaze = [str(u) for u in gaze.numpy()]
                        label = [str(u) for u in labels.cpu().numpy()[k]]
                        name = "example"
                        
                        log = [name] + gaze + label + [str(acc)]
                        
                        outfile.write(",".join(log) + "\n")
                SE_log.close()
                loger = f"[{save_iter}] Total Num: {count}, avg: {total/count} \n"
                outfile.write(loger)
                epoch_log.write(loger)
                print(loger)

