from libraries import *
import baseFunctions as bf



class inception_resnet_v2_regr(nn.Module):

    def __init__(self, device):

        super(inception_resnet_v2_regr, self).__init__()

        self.device = device
        
        self.inception = timm.create_model('inception_resnet_v2', pretrained=True)


        self.avgPooling = nn.AvgPool2d((3,11))    

        self.linear1 = nn.Linear(1536, 1024)
        self.relu1 = nn.ReLU()
        self.batchNorm_linear1 = nn.BatchNorm1d(1024)
        
        self.linear2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.batchNorm_linear2 = nn.BatchNorm1d(256)
        
        self.linear3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.batchNorm_linear3 = nn.BatchNorm1d(64)
        
        self.linear4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()
        self.batchNorm_linear4 = nn.BatchNorm1d(16)

        self.linear5 = nn.Linear(16, 1) #steering angle





    def forward(self, x):
        x = self.inception.forward_features(x)
        
        x = self.avgPooling(x).view(x.shape[0], 1536)
        
        x = self.batchNorm_linear1(self.relu1(self.linear1(x)))
        x = self.batchNorm_linear2(self.relu2(self.linear2(x)))
        x = self.batchNorm_linear3(self.relu3(self.linear3(x)))
        x = self.batchNorm_linear4(self.relu4(self.linear4(x)))
        
        x = self.linear5(x)

        return x
     
    
    def loss(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)
    
    def MeanAbsoluteError(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target)
    
    
"""
def loss(self, pred, target):
    
    weights = bf.weight_fun(bf.normalize_steering(torch.abs(bf.reverse_normalized_steering(target))))
    
    return torch.mean(weights * (pred - target) ** 2)

def MeanAbsoluteError(self, pred, target):
    return torch.nn.functional.l1_loss(pred.reshape(-1), target)
"""



class Trainer():
    
    def __init__(self, model, ckp_dir = "", score_dir = "", score_file = "score.pkl"):
        self.model = model
        self.ckp_dir = ckp_dir
        self.score_dir= score_dir
        self.score_file = score_file
        
    def train_model(self, data, max_epoch=40, steps_per_epoch=0, lr = 1e-3, lr_cap = 1e-3, gamma = 0.5, weight_decay = 0, ckp_save_step=20, log_step=5, ckp_epoch=0):

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

                
                pred = self.model(batch["img"].to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)

                loss = self.model.loss(pred.reshape(-1), gt_steeringAngle)
                
                with torch.no_grad():
                    train_tot_loss += loss * batch['statistics'].shape[0]
                    mae += self.model.MeanAbsoluteError(bf.reverse_normalized_steering(pred.reshape(-1)), bf.reverse_normalized_steering(gt_steeringAngle)) * batch['statistics'].shape[0]

                loss.backward()
                optim.step()

                
                
                if(steps_per_epoch == id_b):
                    self.data.sampler.reset_sampler()
                    break
            
            self.data.sampler.reset_sampler()
                
            if(scheduler.get_last_lr()[0] > lr_cap):
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
        rmse=0
        
        preds = np.array([0])
        
        for id_b, batch in tqdm(enumerate(test_data), total=len(test_data)):
            

            with torch.no_grad():
                
                pred = self.model(batch["img"].to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)

                loss = self.model.loss(pred.reshape(-1), gt_steeringAngle)
                
                
                test_tot_loss += loss
                mae += self.model.MeanAbsoluteError(bf.reverse_normalized_steering(pred.reshape(-1)), bf.reverse_normalized_steering(gt_steeringAngle))
                rmse = torch.sqrt(torch.nn.functional.mse_loss(bf.reverse_normalized_steering(pred.reshape(-1)), bf.reverse_normalized_steering(gt_steeringAngle)))
                
                preds = np.concatenate([preds, pred.cpu().numpy().flatten()])
                
                
        print('Total Test Loss: %7.4f --- MAE: %7.4f --- --- RMSE: %7.4f' % (test_tot_loss/len(test_data), mae/len(test_data), rmse/len(test_data)))
                
        return test_tot_loss/len(test_data), mae/len(test_data), rmse/len(test_data), preds
