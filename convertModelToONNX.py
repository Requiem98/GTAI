from libraries import *
import baseFunctions as bf
import torch.onnx
from models import CNN


if __name__ == '__main__':
    
    #torch.multiprocessing.set_start_method('spawn')
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()


cnn = CNN(device = device).to(device)

cnn.load_state_dict(torch.load("./Data/models/CNN/checkpoint/00020.pth"))

cnn.eval()

x = torch.randn(1, 3, 480, 800, requires_grad=True).to(device)

cnn(x)

# Export the model
torch.onnx.export(cnn,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./Data/models/CNN/CNN.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['image'],   # the model's input names
                  output_names = ['steeringAngle']) # the model's output names