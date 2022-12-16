from libraries import *
import baseFunctions as bf
import torch.onnx
from models import CNN, inception_resnet_v2_regr, NVIDIA


if __name__ == '__main__':
    
    #torch.multiprocessing.set_start_method('spawn')
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()


    #cnn = CNN(device = device).to(device)
    #cnn.load_state_dict(torch.load("./Data/models/CNN/checkpoint/00020.pth"))
    #cnn.eval()
    
    #inception = inception_resnet_v2_regr(device = device).to(device)
    #inception.load_state_dict(torch.load("./Data/models/Inception/checkpoint/00005.pth"))
    #inception.eval()
    
    
    nvidia = NVIDIA(device = device).to(device) #qui inserire modello da trainare
    nvidia.load_state_dict(torch.load("./Data/models/NVIDIA/checkpoint/00125.pth"))
    nvidia.eval()
    
    dummy_input = torch.randn(1, 3, 240, 400, requires_grad=True).to(device)
    
    nvidia(dummy_input)
    
    # Export the model
    torch.onnx.export(nvidia,               # model being run
                      dummy_input,                         # model input (or a tuple for multiple inputs)
                      "./Data/models/NVIDIA/nvidia.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=16,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['image'],   # the model's input names
                      output_names = ['steeringAngle']) # the model's output names
    
    
    
    
    

