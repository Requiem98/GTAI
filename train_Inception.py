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
    
    train_dataset = bf.GTADataset("data_train_norm.csv", DATA_ROOT_DIR, bf.preprocess)
    test_dataset = bf.GTADataset("data_test_norm.csv", DATA_ROOT_DIR, bf.test_preprocess)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=32, 
                            sampler=bf.SteeringSampler("./Data/data_train_norm.csv"), 
                            num_workers=10)

    
    test_dl = DataLoader(test_dataset, 
                            batch_size=32, 
                            num_workers=10)

    

 
    inception = inception_resnet_v2_regr(device = device).to(device)
    inception.load_state_dict(torch.load("./Data/models/Inception/checkpoint/00008.pth"))
    
    
    trainer = Trainer(inception, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)
    
    """
    trainer.train_model(train_dl,
                        max_epoch=10, 
                        steps_per_epoch=0,
                        lr=0.01,
                        gamma = 0.8,
                        weight_decay=1e-6,
                        log_step=1, 
                        ckp_save_step = 2,
                        ckp_epoch=0)
    """
    print('Starting test...')
    test_tot_loss, mae, rmse, o = trainer.test_model(test_dl)
    
    data = pd.read_csv(DATA_ROOT_DIR + 'data_test_norm.csv', index_col=0)
    
    a=10000
    plt.plot(bf.reverse_normalized_steering(o[1:a]))
    plt.plot(np.arange(a), data["steeringAngle"][:a], alpha=0.5)
    
    
    
    results = pd.DataFrame(np.array([["Inception"], [test_tot_loss.cpu().numpy()], [mae.cpu().numpy()], [rmse.cpu().numpy()]]).reshape(-1,4), columns=["model_name","test_tot_loss", "mae", "rmse"])
    
    results_df = pd.read_csv("./Data/results.csv", index_col=0)
    
    results_df = pd.concat([results_df, results])
    
    results_df.to_csv("./Data/results.csv")

#== Best Result ==



