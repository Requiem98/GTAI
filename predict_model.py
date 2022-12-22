from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import mss
import win32gui, win32api
import time
import PIL
import pyvjoy
import cv2
from libraries import *
import baseFunctions as bf

from directkeys import PressKey
from directkeys import ReleaseKey
from PIL import Image

from models import MapNet



j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768


def preprocess_frame(image):
    image = image.crop(box=(0, 200, 800, 480))
    image = F.resize(image,(140,400))
    image = F.to_tensor(image)            
    image = image.unsqueeze(0)
    
    return image

def preprocess_frame_and_map(image):
    
    #mmap = image[520:580,56:116]
    mmap = image.crop(box=(56, 520, 116, 580))
    mmap = F.to_tensor(mmap)
    mmap = mmap.unsqueeze(0)
    
    image = image.crop(box=(0, 200, 800, 480))
    image = F.resize(image,(140,400))
    image = F.to_tensor(image)            
    image = image.unsqueeze(0)
    
    return image, mmap

class FPSTimer:
	def __init__(self):
		self.t = time.time()
		self.iter = 0
		
	def reset(self):
		self.t = time.time()
		self.iter = 0
		
	def on_frame(self):
		self.iter += 1
		if self.iter == 100:
			e = time.time()
			print('FPS: %0.2f' % (100.0 / (e - self.t)))
			self.t = time.time()
			self.iter = 0



def lerp(a, b, t):
	return (t * a) + ((1-t) * b)
		
def predict_loop(device):
    
    model = MapNet(device = device).to(device) #qui inserire modello da trainare
    model.load_state_dict(torch.load("./Data/models/MapNet/checkpoint/00015.pth"))
    model.eval()
    
    timer = FPSTimer()
	
    pause=True
    return_was_down=False
	

    sct = mss.mss()
    
    left = int((2560 - 800)/2)
    top = int((1440 - 600)/2)
    
    mon = {'top': top, 'left': left, 'width': 800, 'height': 600}
	
    i=0
	
    
    print('Ready')
	
    while True:
        i += 1
        if (win32api.GetAsyncKeyState(0x08)&0x8001 > 0):
            print("stop!")
            break
		
        if (win32api.GetAsyncKeyState(0x0D)&0x8001 > 0):
            if (return_was_down == False):
                if (pause == False):
                    pause = True
					
                    j.data.wAxisX = int(vjoy_max * 0.5)
                    j.data.wAxisY = int(vjoy_max * 0)
                    j.data.wAxisZ = int(vjoy_max * 0)

                    j.update()
					
                    print('Paused')
                else:
                    pause = False
					
                    print('Resumed')
				
            return_was_down = True
        else:
            return_was_down = False
		
        if (pause):
            time.sleep(0.01)
            continue
        


        sct_img = sct.grab(mon)
        image = Image.frombytes('RGB', sct_img.size, sct_img.rgb)

        
        image, mmap = preprocess_frame_and_map(image)
        
                
		
        
		
        steeringAngle = model(image.to(device), mmap.to(device))

        steeringAngle = steeringAngle.cpu().detach().numpy()[0].item()

        steeringAngle = ((steeringAngle+1)/2.0)
		
        j.data.wAxisX = int(vjoy_max * min(max(steeringAngle, 0), 1))
        
        if(i%2==0):
            j.data.wAxisY = int(vjoy_max * 0.5)
            j.data.wAxisZ = int(vjoy_max * 0)
        else:
            j.data.wAxisY = int(vjoy_max * 0)
            j.data.wAxisZ = int(vjoy_max * 0)
		
        j.update()
		
        os.system('cls')
        print("Steering Angle: %.2f" % min(max(steeringAngle, 0), 1))
        print("Throttle: %.2f" % min(max(0.5, 0), 1))
        print("Brake: %.2f" % min(max(0, 0), 1))
		
		#timer.on_frame()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()
        
    predict_loop(device)
