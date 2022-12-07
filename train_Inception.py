from libraries import *
import baseFunctions as bf
from models import inception_resnet_v2_regr, Trainer


if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()
        
    
    CKP_DIR = "./Data/models/Inception/checkpoint/"
    SCORE_DIR = "./Data/models/Inception/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    
    train_dataset = bf.GTADataset("data.csv", DATA_ROOT_DIR, bf.preprocess)
    
    dataloader = DataLoader(train_dataset, 
                            batch_size=6, 
                            sampler=bf.SteeringSampler("./Data/data.csv"), 
                            num_workers=20, 
                            prefetch_factor = 2)
    

 
    inception = inception_resnet_v2_regr(device = device).to(device)
    
    
    trainer = Trainer(inception, dataloader, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)

    
    trainer.train_model(max_epoch=5, 
                        steps_per_epoch=0,
                        lr=0.01,
                        gamma = 0.8,
                        weight_decay=1e-6,
                        log_step=1, 
                        ckp_save_step = 1,
                        ckp_epoch=5)

    

#== Best Result ==
#Current Learning Rate:  0.0006  --- Total Train Loss: 32.9991  22 epochs with steps_per_epoch = 2000
#Current Learning Rate: 0.00328 --- Total Train Loss:  0.0032 --- MAE:  0.1848 5 epoche with steps_per_epoch = 0

