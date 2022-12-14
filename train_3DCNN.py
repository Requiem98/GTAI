from libraries import *
from models import CNN
import baseFunctions as bf

#bf.create_train_test_dataframe(data, group_n=2, test_size=0.2, test_file_name = "data_test_sequential.csv", train_file_name = "data_train_sequential.csv")


class Trainer():
    
    def __init__(self, model, data, ckp_dir = "", score_dir = "", score_file = "score.pkl"):
        self.model = model
        self.data = data
        self.ckp_dir = ckp_dir
        self.score_dir= score_dir
        self.score_file = score_file
        
    def train_model(self, max_epoch=40, steps_per_epoch=0, lr = 1e-3, gamma = 0.5, weight_decay = 0, ckp_save_step=20, log_step=5, ckp_epoch=0):

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

                #(B, 2, C, H, W)
                #(B, C, H, W)
                b = batch["img"][:,1] - batch["img"][:,0]
            

                pred = self.model(b.to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,1,0].to(self.model.device)

                loss = self.model.weighted_mse_loss(pred.reshape(-1), gt_steeringAngle)
                
                with torch.no_grad():
                    train_tot_loss += loss * batch['statistics'].shape[0]
                    mae += self.model.MeanAbsoluteError(pred, gt_steeringAngle) * batch['statistics'].shape[0]

                loss.backward()
                optim.step()
                
                
                
                if(steps_per_epoch == id_b):
                    break
            
            
                
            if(scheduler.get_last_lr()[0] > 1e-4):
                pass
                #scheduler.step()


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
        
        for id_b, batch in tqdm(enumerate(test_data), total=len(test_data)):
            

            with torch.no_grad():
                
                b = batch["img"][:,0] - batch["img"][:,1]
                
                pred = self.model(b.to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,1,0].to(self.model.device)

                loss = self.model.weighted_mse_loss(pred.reshape(-1), gt_steeringAngle)
                
                
                test_tot_loss += loss * batch['statistics'].shape[0]
                mae += self.model.MeanAbsoluteError(pred, gt_steeringAngle) * batch['statistics'].shape[0]
                
        print('Total Test Loss: %7.4f --- MAE: %7.4f' % (test_tot_loss/len(test_data), mae/len(test_data)))
                
        return test_tot_loss/len(test_data), mae/len(test_data)


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
    CKP_DIR = "./Data/models/DiffCNN/checkpoint/"
    SCORE_DIR = "./Data/models/DiffCNN/scores/"
    SCORE_FILE = 'history_score.pkl'
    
        
    train_dataset = bf.GTADataset("data_train_sequential.csv", DATA_ROOT_DIR, bf.preprocess)
    test_dataset = bf.GTADataset("data_test_sequential.csv", DATA_ROOT_DIR, bf.preprocess)
    
    train_dl = DataLoader(train_dataset, 
                          batch_size=256,
                          sampler=SubsetRandomSampler(list(BatchSampler(SequentialSampler(train_dataset), 2, drop_last=True))),
                          num_workers=10,
                          prefetch_factor=8)

    
    test_dl = DataLoader(test_dataset,
                         batch_size=256,
                         sampler=SubsetRandomSampler(list(BatchSampler(SequentialSampler(train_dataset), 2, drop_last=True))),
                         num_workers=10,
                         prefetch_factor=4)




    diff_cnn = CNN(device = device).to(device) #qui inserire modello da trainare
    

    trainer = Trainer(diff_cnn, train_dl, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)

    
    trainer.train_model(max_epoch=10, 
                        steps_per_epoch=0,
                        lr=0.01,
                        gamma = 0.8,
                        weight_decay=1e-6,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=10)
    
    
    print('Starting test...')
    trainer.test_model(test_dl)
    

#== Best Result ==

