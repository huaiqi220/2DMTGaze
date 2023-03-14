from model import load_model
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import copy
import yaml
import math
import time
import importlib
from tensorboardX import SummaryWriter
import string

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device_ids = [1,5,7]
    config = yaml.load(open("config/aff_config.yaml"), Loader=yaml.FullLoader)
    readername = config["reader"]
    dataloader = importlib.import_module("reader-dir." + readername)
    config = config["train"]
    model_config = config['model']
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["model"]["model_name"]
    now_train = str(modelname) + "/bs_"  + str(config["params"]["batch_size"])  + "_ep_" + str(str(config["params"]["epoch"])) + "_lr_" + str(str(config["params"]["lr"])) \
         +"_1"
    writer = SummaryWriter('runs/' + now_train)
    step_number = 0

    trains = os.listdir(labelpath)
    trains.sort()
    print(f"Train Sets Num:{len(trains)}")
    trainlabelpath = [os.path.join(labelpath, j) for j in trains] 
    save_path = os.path.join(config["save"]["save_path"],"checkpoint", now_train)


    # save_path = os.path.join(config["save"]["save_path"], "checkpoint")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda:"+ str(device_ids[0]) if torch.cuda.is_available() else "cpu")

    # print("Read data")
    # dataset = reader.txtload(path, "train", config["params"]["batch_size"], shuffle=True,
    #                          num_workers=0)
    print("Read data")
    print(config["params"]["batch_size"])
    print(config["params"]["image_size"])

    dataset = dataloader.txtload(trainlabelpath, imagepath, config["params"]["batch_size"], image_size=config["params"]["image_size"], shuffle=True, num_workers=32, header=True)

    print("Model building")
    net = load_model.loadTheModel(model_config)

    net.train()
    net = nn.DataParallel(net,device_ids)
    # state_dict = torch.load(os.path.join("/disk1/repository/gazeCheckpoint/Gaze-tf/checkpoint/new_model/bs_256_ep_20_lr_0.0001_1", f"Iter_20_new_model.pt"))
    # net.load_state_dict(state_dict)
    net.to(device)

    # for name, value, in net.named_parameters():
    #     print(name)
    #     if name == "module.fc.0.weight" or name == "module.fc.0.bias" or name == "module.fc.2.weight" or name == "module.fc.2.bias":
    #         value.requires_grad=True
    #     else:
    #         value.requires_grad=False

    
    # cur_params = filter(lambda p: p.requires_grad, net.parameters())  
    cur_params =  net.parameters()


    print("optimizer building")
    loss_op = nn.SmoothL1Loss().cuda()
    base_lr = config["params"]["lr"]
    cur_step = 0
    decay_steps = config["params"]["decay_step"]
    optimizer = torch.optim.Adam(cur_params, base_lr,
                                weight_decay=0.0005)
    print("Traning")
    length = len(dataset)
    cur_decay_index = 0
    with open(os.path.join(save_path, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            if cur_decay_index < len(decay_steps) and epoch == decay_steps[cur_decay_index]:
                base_lr = base_lr * config["params"]["decay"]
                cur_decay_index = cur_decay_index + 1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr
            #if (epoch <= 10):
            #    continue

            time_begin = time.time()
            # print("start" + str(time.time())) 
            for i, (data) in enumerate(dataset):
                # print("the" + str(i) + " batch data load time:" + str(time.time()))
                batch = {}
                batch["faceImg"] = data["face"].to(device)
                batch["leftEyeImg"] = data["left"].to(device)
                batch['rightEyeImg'] = data['right'].to(device)
                batch['rects'] = data['rects'].to(device)
                batch['origin'] = data['full'].to(device)
                label = data["label"].to(device)
                # print("the" + str(i) + " batch forward start time:" + str(time.time()))
                gaze = net(batch, device)
                # print("the" + str(i) + " batch forward finish time:" + str(time.time()))
                loss = loss_op(gaze, label)*4
                optimizer.zero_grad()
                loss.backward()
                # print("the" + str(i) + " batch backward finish time:" + str(time.time()))
                optimizer.step()
                time_remain = (length-i-1) * ((time.time()-time_begin)/(i+1)) /  3600   #time estimation for current epoch
                epoch_time = (length-1) * ((time.time()-time_begin)/(i+1)) / 3600       #time estimation for 1 epoch
                #person_time = epoch_time * (config["params"]["epoch"])                  #time estimation for 1 subject
                time_remain_total = time_remain + \
                                    epoch_time * (config["params"]["epoch"]-epoch)
                                    #person_time * (len(subjects) - subject_i - 1) 
                writer.add_scalar('loss',loss,global_step=step_number)
                step_number = step_number + 1
                log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss:.5f} lr:{base_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                # if i % 20 == 0:
                print(log)
                sys.stdout.flush()
                outfile.flush()

            if epoch % config["save"]["step"] == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{modelname}.pt"))

