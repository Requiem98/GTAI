from libraries import *
import baseFunctions as bf
from models import Inception_MapResNet



class Trainer():
    
    def __init__(self, model, ckp_dir = "", score_dir = "", score_file = "score.pkl"):
        self.model = model
        self.ckp_dir = ckp_dir
        self.score_dir= score_dir
        self.score_file = score_file
        
    def train_model(self, data, max_epoch=40, steps_per_epoch=0, lr = 1e-3, gamma = 0.5, weight_decay = 0, ckp_save_step=20, log_step=5, ckp_epoch=0):

       # Argument for the training
       #max_epoch          # Total number of epoch
       #ckp_save_step      # Frequency for saving the model
       #log_step           # Frequency for printing the loss
       #lr                 # Learning rate
       #weight_decay       # Weight decay
       #ckp_dir            # Directory where to save the checkpoints
       #score_dir          # Directory where to save the scores
       #score_file         # Name of the scores file
       #ckp_epoch          # If the checkpoint file is passed, this indicate the checkpoint training epoch
       #ckp_epoch          # Load weights from indicated epoch if a corresponding checkpoint file is present
       
        self.data = data
       
        if(steps_per_epoch==0):
            steps_per_epoch=len(self.data)
       
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)

        if(ckp_epoch != 0):
            self.model.load_state_dict(torch.load(self.ckp_dir + f'{(ckp_epoch):05d}.pth'))
            optim.load_state_dict(torch.load(self.ckp_dir + f'optim_{(ckp_epoch):05d}.pth'))
            history_score = bf.read_object(self.score_dir + f'{(ckp_epoch):05d}_' + self.score_file)
        else:
            history_score = defaultdict(list)
            
       
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma, last_epoch= ckp_epoch-1, verbose=False)
        #scaler = torch.cuda.amp.GradScaler()
            
        torch.backends.cudnn.benchmark = True
        
        # compute execution time of the cell
        start_time = time.time()




        print("Start Training...\n")


        for epoch in range(max_epoch):

            if (epoch+1) % log_step == 0:
                print("---> Epoch %03i/%03i <--- " % ((epoch+1), max_epoch))

            ###### TRAIN ######
            self.model.train()

            train_tot_loss = 0
            mae = 0

            for id_b, batch in tqdm(enumerate(self.data), total=steps_per_epoch):
                
                optim.zero_grad()

                
                pred = self.model(batch["img"].to(self.model.device), batch["mmap"].to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)

                loss = self.model.loss(pred.reshape(-1), gt_steeringAngle)
                
                with torch.no_grad():
                    train_tot_loss += loss * batch['statistics'].shape[0]
                    mae += self.model.MeanAbsoluteError(bf.reverse_normalized_steering(pred.reshape(-1)), bf.reverse_normalized_steering(gt_steeringAngle)) * batch['statistics'].shape[0]

                loss.backward()
                optim.step()

                
                
                if(steps_per_epoch == id_b):
                    #self.data.sampler.reset_sampler()
                    break
            
            #self.data.sampler.reset_sampler()
                
            if(scheduler.get_last_lr()[0] > 1e-4):
                scheduler.step()


            if (epoch+1) % log_step == 0:
                print('Current Learning Rate: %7.5f --- Total Train Loss: %7.4f --- MAE: %7.4f' % (scheduler.get_last_lr()[0], train_tot_loss/steps_per_epoch, mae/steps_per_epoch))


            history_score['loss_tot_train'].append((train_tot_loss/steps_per_epoch).item())
            history_score['MAE_train'].append((mae/steps_per_epoch).item())


            # Here we save checkpoints to avoid repeated training
            if ((epoch+1) % (ckp_save_step) == 0):
                print("Saving checkpoint... \n ")
                torch.save(self.model.state_dict(), self.ckp_dir + f'{(epoch+1+ckp_epoch):05d}.pth')
                torch.save(optim.state_dict(), self.ckp_dir + f'optim_{(epoch+1+ckp_epoch):05d}.pth')
                bf.save_object(history_score, self.score_dir + f'{(epoch+1+ckp_epoch):05d}_' + self.score_file)
                


        print("Saving checkpoint... \n ")
        torch.save(self.model.state_dict(), self.ckp_dir + f'{(epoch+1+ckp_epoch):05d}.pth')
        torch.save(optim.state_dict(), self.ckp_dir + f'optim_{(epoch+1+ckp_epoch):05d}.pth')
        bf.save_object(history_score, self.score_dir + f'{(epoch+1+ckp_epoch):05d}_' + self.score_file)

        # print execution time
        print("Total time: %s seconds" % (time.time() - start_time))
        
        
    def test_model(self, test_data):
        
        self.model.eval()
        
        test_tot_loss=0
        mae=0
        
        preds = np.array([0])
        
        for id_b, batch in tqdm(enumerate(test_data), total=len(test_data)):
            

            with torch.no_grad():
                
                pred = self.model(batch["img"].to(self.model.device), batch["mmap"].to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)

                loss = self.model.loss(pred.reshape(-1), gt_steeringAngle)
                
                
                test_tot_loss += loss * batch['statistics'].shape[0]
                mae += self.model.MeanAbsoluteError(bf.reverse_normalized_steering(pred.reshape(-1)), bf.reverse_normalized_steering(gt_steeringAngle)) * batch['statistics'].shape[0]
                
                preds = np.concatenate([preds, pred.cpu().numpy().flatten()])
                
        print('Total Test Loss: %7.4f --- MAE: %7.4f' % (test_tot_loss/len(test_data), mae/len(test_data)))
                
        return test_tot_loss/len(test_data), mae/len(test_data), preds



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
    CKP_DIR = "./Data/models/Inception_MapResNet/checkpoint/"
    SCORE_DIR = "./Data/models/Inception_MapResNet/scores/"
    SCORE_FILE = 'history_score.pkl'
    
        
    train_dataset = bf.GTADataset("data_train_norm.csv", DATA_ROOT_DIR, bf.preprocess, mmap=True)
    test_dataset = bf.GTADataset("data_test_norm.csv", DATA_ROOT_DIR, bf.test_preprocess, mmap=True)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=32, 
                            shuffle=True,
                            #sampler=bf.SteeringSampler("./Data/data_train_norm.csv"), 
                            num_workers=10)

    
    test_dl = DataLoader(test_dataset, 
                            batch_size=32, 
                            num_workers=10)


    mapnet = Inception_MapResNet(device = device).to(device) #qui inserire modello da trainare
    mapnet.load_state_dict(torch.load("./Data/models/Inception_MapResNet/checkpoint/00005.pth"))
    
    
    trainer = Trainer(mapnet, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)


    trainer.train_model(train_dl,
                        max_epoch=1, 
                        steps_per_epoch=2,
                        lr=0.01,
                        gamma = 0.8,
                        weight_decay=1e-6,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=0)
 
    print('Starting test...')
    _, _, o = trainer.test_model(test_dl)
    
   
    data = pd.read_csv(DATA_ROOT_DIR + 'data_test_norm.csv', index_col=0)
    
    a=10000
    plt.plot(bf.reverse_normalized_steering(o[1:a]))
    plt.plot(np.arange(a), data["steeringAngle"][:a], alpha=0.5)
    
    
   
    inp = np.array(bf.reverse_normalized_steering(o))[1:]
    targ = data["steeringAngle"].to_numpy()
    
    (np.abs(inp - targ)).mean()
    np.sqrt(((inp - targ)**2).mean())

    torch.nn.functional.l1_loss(torch.tensor(inp), torch.tensor(targ))
    

    
    
#== Best Result ==
#Total Test Loss:  0.0909 --- MAE:  7.8666




