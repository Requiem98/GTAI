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
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    cnn = CNN(device=device).to(device)
      
    
    
    cnn.train_model(dataloader, 
                    max_epoch=1, 
                    #lr=5e-3, 
                    log_step=1, 
                    ckp_save_step = 5, 
                    ckp_dir = CKP_DIR, 
                    score_dir = SCORE_DIR, 
                    score_file = SCORE_FILE)


