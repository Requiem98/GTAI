from libraries import *
import baseFunctions as bf
from models import CNN, Trainer


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
    
    train_dataset = bf.GTADataset("data.csv", DATA_ROOT_DIR, bf.preprocess)
    
    dataloader = DataLoader(train_dataset, 
                            batch_size=128, 
                            sampler=bf.SteeringSampler("./Data/data.csv"), 
                            num_workers=20, 
                            prefetch_factor = 2)


    cnn = CNN(device = device).to(device)
    
    
    trainer = Trainer(cnn, dataloader, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)

    
    trainer.train_model(max_epoch=5, 
                        steps_per_epoch=0,
                        lr=0.01,
                        gamma = 0.8,
                        weight_decay=1e-6,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=25)
    

#== Best Result ==
#Current Learning Rate:  0.0001 --- Total Train Loss:  1.8021 --- MAE:  4.4562 epoch=20
#Current Learning Rate:  0.0000 --- Total Train Loss:  0.7356 --- MAE:  4.0525 epoch=25