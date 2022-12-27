from libraries import *
import baseFunctions as bf


class NVIDIA(nn.Module):

    def __init__(self, device):

        super(NVIDIA, self).__init__()

        self.device = device

        
        self.conv1 = nn.Conv2d(3, 24, 5, 2, 0)
        self.convlRelu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(24)
        
        self.conv2 = nn.Conv2d(24, 36, 5, 2, 0)
        self.convlRelu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(36)
        
        self.conv3 = nn.Conv2d(36, 48, 5, 2, 0)
        self.convlRelu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm2d(48)
        
        
        self.conv4 = nn.Conv2d(48, 64, 3, 1, 0)
        self.convlRelu4 = nn.ReLU()
        self.batchNorm4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)
        self.convlRelu5 = nn.ReLU()
        self.batchNorm5 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(27520, 1164)
        self.lRelu1 = nn.ReLU()
        self.batchNorm_linear1 = nn.BatchNorm1d(1164)

        self.linear_2 = nn.Linear(1164, 200)
        self.lRelu2 = nn.ReLU()
        self.batchNorm_linear2 = nn.BatchNorm1d(200)

        self.linear_3 = nn.Linear(200, 50)
        self.lRelu3 = nn.ReLU()
        self.batchNorm_linear3 = nn.BatchNorm1d(50)
        
        self.linear_4 = nn.Linear(50, 10)
        self.lRelu4 = nn.ReLU()
        self.batchNorm_linear4 = nn.BatchNorm1d(10)

        self.linear_5 = nn.Linear(10, 1) #steering angle





    def forward(self, x):
        x = self.batchNorm1(self.convlRelu1(self.conv1(x)))
        x = self.batchNorm2(self.convlRelu2(self.conv2(x)))
        x = self.batchNorm3(self.convlRelu3(self.conv3(x)))

        x = self.batchNorm4(self.convlRelu4(self.conv4(x)))
        x = self.batchNorm5(self.convlRelu5(self.conv5(x)))

        x = self.flatten(x)

        x = self.batchNorm_linear1(self.lRelu1(self.linear_1(x)))
        x = self.batchNorm_linear2(self.lRelu2(self.linear_2(x)))
        x = self.batchNorm_linear3(self.lRelu3(self.linear_3(x)))
        x = self.batchNorm_linear4(self.lRelu4(self.linear_4(x)))
        
        x = self.linear_5(x)

        return x
    
    
    def loss(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)
    
    def MeanAbsoluteError(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target)
  









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