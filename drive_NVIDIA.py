#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from libraries import *
import baseFunctions as bf
from models import NVIDIA
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client

import argparse
import time
import cv2

class Model:
    def run(self,frame):
        return [1.0, 0.0, 0.0] # throttle, brake, steering

# Controls the DeepGTAV vehicle
if __name__ == '__main__':
    
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port. 
    # If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
    # We don't want to save a dataset in this case
    #client = Client(ip=args.host, port=args.port)
    client = Client()
    
    # We set the scenario to be in manual driving, and everything else random (time, weather and location). 
    # See deepgtav/messages.py to see what options are supported
    scenario = Scenario(location=[-2573.13916015625, 3292.256103515625, 13.241103172302246], 
                        weather="SUNNY", time=[9,10], vehicle = "voltic", drivingMode=-1) #manual driving
    
    # Send the Start request to DeepGTAV. Dataset is set as default, we only receive frames at 10Hz (320, 160)
    client.sendMessage(Start(scenario=scenario, dataset=Dataset(frame=[800,600])))
    
    #model = Model()
    model = NVIDIA(device=device).to(device)
    model.load_state_dict(torch.load("./Data/models/NVIDIA/checkpoint/00125.pth"))
    model.eval()
    images = list()
    # Start listening for messages coming from DeepGTAV. We do it for 80 hours
    stoptime = time.time() + 80*3600
    while time.time() < stoptime:
        try:
            # We receive a message as a Python dictionary
            message = client.recvMessage()  
                
            # The frame is a numpy array that can we pass through a CNN for example     
            image = frame2numpy(message['frame'], (800,600))
            
            image = F.to_pil_image(image[..., ::-1])
            image = F.resize(image,(240,400))
            images.append(image)
            image = F.to_tensor(image)            
            image = image.unsqueeze(0)
            
            steeringAngle = model(image.to(device))
            #commands = model.run(image)
            
            steeringAngle = steeringAngle.cpu().detach().numpy()[0].item()
            
            print(steeringAngle)
            
            # We send the commands predicted by the agent back to DeepGTAV to control the vehicle
            client.sendMessage(Commands(0.3, 0.0, steeringAngle))
        except KeyboardInterrupt:
            break
            
    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()




len(images)


plt.imshow(np.array(images[30]))











