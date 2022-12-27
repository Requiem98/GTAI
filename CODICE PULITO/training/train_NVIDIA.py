from libraries import *
import baseFunctions as bf
from models import NVIDIA, Trainer


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
    CKP_DIR = "./Data/models/NVIDIA/checkpoint/"
    SCORE_DIR = "./Data/models/NVIDIA/scores/"
    SCORE_FILE = 'history_score.pkl'
    
        
    train_dataset = bf.GTADataset("data_train_norm.csv", DATA_ROOT_DIR, bf.preprocess)
    val_dataset = bf.GTADataset("data_test_norm.csv", DATA_ROOT_DIR, bf.test_preprocess)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=512, 
                            sampler=bf.SteeringSampler(train_dataset), 
                            num_workers=10)

    
    val_dl = DataLoader(test_dataset, 
                            batch_size=512, 
                            num_workers=10)


    nvidia = NVIDIA(device = device).to(device) #qui inserire modello da trainare
    
    
    trainer = Trainer(nvidia, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)
    
    
    trainer.train_model(train_dl,
                        max_epoch=2, 
                        steps_per_epoch=100,
                        lr=0.01,
                        gamma = 0.8,
                        weight_decay=0,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=0)
    
    
    print('Starting validation...')
    test_tot_loss, mae, rmse, o1 = trainer.test_model(val_dl)
    
   
    val_dataframe = pd.read_csv(DATA_ROOT_DIR + 'data_test_norm.csv', index_col=0)
    
    a=100
    plt.plot(bf.reverse_normalized_steering(o1[1:a]))
    plt.plot(np.arange(a), val_dataframe["steeringAngle"][:a], alpha=0.5)
    


#== Best Result ==
#Total Test Loss:  0.0290 --- MAE:  4.0833 --- --- RMSE:  0.0815




