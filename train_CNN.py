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
        
    
    #Path per i salvataggi dei checkpoints
    #SINTASSI: ./Data/models/NOME_MODELLO/etc...
    CKP_DIR = "./Data/models/CNN/checkpoint/"
    SCORE_DIR = "./Data/models/CNN/scores/"
    SCORE_FILE = 'history_score.pkl'
    
        
    train_dataset = bf.GTADataset("data_train_norm.csv", DATA_ROOT_DIR, bf.preprocess)
    test_dataset = bf.GTADataset("data_test_norm.csv", DATA_ROOT_DIR, bf.preprocess)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=256, 
                            sampler=bf.SteeringSampler("./Data/data_train_norm.csv"), 
                            num_workers=10)

    
    test_dl = DataLoader(test_dataset, 
                            batch_size=256, 
                            num_workers=10)


    cnn = CNN(device = device).to(device) #qui inserire modello da trainare
    #cnn.load_state_dict(torch.load("./Data/models/CNN/checkpoint/00015.pth"))
    
    
    trainer = Trainer(cnn, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)

    trainer.train_model(train_dl,
                        max_epoch=35, 
                        steps_per_epoch=0,
                        lr=0.01,
                        gamma = 0.8,
                        weight_decay=1e-6,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=20)

    print('Starting test...')
    _, _, o = trainer.test_model(test_dl)
    
    
    data = pd.read_csv(DATA_ROOT_DIR + 'data_test_norm.csv', index_col=0)
    
    a=10000
    plt.plot(bf.reverse_normalized_steering(o[1:a]))
    plt.plot(np.arange(a), data["steeringAngle"][:a], alpha=0.5)
    
    

    

#== Best Result ==
#Total Test Loss:  0.0909 --- MAE:  7.8666




