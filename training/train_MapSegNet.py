from libraries import *
import baseFunctions as bf
from torchvision.transforms import functional as f
from models.MapSegNet import *


if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()
        
    
    #Path per i salvataggi dei checkpoints
    #SINTASSI: ./Data/models/NOME_MODELLO/etc...
    CKP_DIR = "./Data/models/MapSegNet/checkpoint/"
    SCORE_DIR = "./Data/models/MapSegNet/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    
    train_dataset = bf.GTADataset(csv_file = "imagesTOT_balanced.csv", root_dir=DATA_ROOT_DIR, transform=bf.preprocess_segment, mmap = True, normalize=False)
    test_dataset  = bf.GTADataset(csv_file = "images3(TEST).csv", root_dir=DATA_ROOT_DIR, transform=bf.preprocess_segment, mmap = True, normalize=False)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=512,
                            shuffle=True,
                            num_workers=10)
    
    
    test_dl = DataLoader(test_dataset, 
                            batch_size=800, 
                            num_workers=0)

    
    map_seg_net = MapSegNet(device = device).to(device) #qui inserire modello da trainare
    #map_seg_net.load_state_dict(torch.load("./Data/models/Unet/checkpoint/00100.pth"))
    
    
    trainer = Trainer(map_seg_net, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)
    
    
    trainer.train_model(train_dl,
                        max_epoch=2, 
                        steps_per_epoch=1,
                        lr=0.01,
                        weight_decay=0,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=0)
   
    
    
    
    
    
    
    
    
    