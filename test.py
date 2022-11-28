from libraries import *
import baseFunctions as bf
from models import CNN
import timm


#if __name__ == '__main__':
if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()
        
    
    CKP_DIR = "./Data/models/CNN/checkpoint/"
    SCORE_DIR = "./Data/models/CNN/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    
    train_dataset = bf.GTADataset("data.csv", DATA_ROOT_DIR, bf.preprocess, load_all=False)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    for batch in dataloader:
        b = batch
        break
    
    
    
    model = timm.create_model('inception_resnet_v2', pretrained=False).to(device)
    
    o = model.forward_features(b["img"].to(device))
    
    
    
    o.shape

    nn.AvgPool2d((9,23))(o).view(o.shape[0], 1536).shape

