from libraries import *
import baseFunctions as bf


class CNN(nn.Module):

    def __init__(self, device):

        super(CNN, self).__init__()

        self.device = device

        #self.norm = nn.LayerNorm()
        self.conv1 = nn.Conv2d(3, 24, 5, 2, 0)
        self.convlRelu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(24, 40, 5, 2, 0)
        self.convlRelu2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(40, 48, 5, 2, 0)
        self.convlRelu3 = nn.LeakyReLU()
        
        
        self.conv4 = nn.Conv2d(48, 64, 3, 1, 0)
        self.convlRelu4 = nn.LeakyReLU()
        
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)
        self.convlRelu5 = nn.LeakyReLU()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(220224, 1200)
        self.lRelu1 = nn.LeakyReLU()

        self.linear2 = nn.Linear(1200, 104)
        self.lRelu2 = nn.LeakyReLU()

        self.linear3 = nn.Linear(104, 48)
        self.lRelu3 = nn.LeakyReLU()

        self.linear4 = nn.Linear(48, 8)
        self.lRelu4 = nn.LeakyReLU()


        self.linear5 = nn.Linear(8, 1) #steering angle





    def forward(self, x):
        x = self.convlRelu1(self.conv1(x))
        x = self.convlRelu2(self.conv2(x))
        x = self.convlRelu3(self.conv3(x))

        x = self.convlRelu4(self.conv4(x))
        x = self.convlRelu5(self.conv5(x))

        x = self.flatten(x)

        x = self.lRelu1(self.linear1(x))
        x = self.lRelu2(self.linear2(x))
        x = self.lRelu3(self.linear3(x))
        x = self.lRelu4(self.linear4(x))
        x = self.linear5(x)

        return x


    def train_model(self, data, max_epoch=40, lr = 1e-3, weight_decay = 0, ckp_save_step=20, log_step=5, ckp_dir = "", score_dir = "", score_file = "score.pkl"):

       # Argument for the training
       #max_epoch          # Total number of epoch
       #ckp_save_step      # Frequency for saving the model
       #log_step           # Frequency for printing the loss



        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        scaler = torch.cuda.amp.GradScaler()
            
        torch.backends.cudnn.benchmark = True
        
        # compute execution time of the cell
        start_time = time.time()



        #Vector to store loss
        history_score = defaultdict(list)

        print("Start Training...\n")


        for epoch in range(max_epoch):

            if (epoch+1) % log_step == 0:
                print("---> Epoch %03i/%03i <--- " % ((epoch+1), max_epoch))

            ###### TRAIN ######
            self.train()

            train_steeringAngle_loss=0
            train_speed_loss = 0
            train_acceleration_loss = 0
            train_tot_loss = 0

            for id_b, batch in tqdm(enumerate(data), total=len(data)):
                # All the gradients are resetted to zero before the training step
                optim.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # We predict target speeds and we save the corresponing GTs
                    pred = self.forward(batch["img"].to(self.device))
                    
    
                    gt_steeringAngle = batch["statistics"][:,0].to(self.device)

    
                    loss = nn.functional.mse_loss(pred.reshape(-1), gt_steeringAngle)

                    train_tot_loss += loss * batch['statistics'].shape[0]

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                


            if (epoch+1) % log_step == 0:
                #print('Total Train Loss: %7.4f - Steering angle Loss: %7.4f - Speed Loss: %7.4f - Acceleration Loss: %7.4f' % (train_tot_loss/len(data), train_steeringAngle_loss/len(data), train_speed_loss/len(data), train_acceleration_loss/len(data)))
                print('Total Train Loss: %7.4f' % (train_tot_loss/len(data)))


            history_score['loss_tot_train'].append((train_tot_loss/len(data)).item())


            # Here we save checkpoints to avoid repeated training
            if ((epoch+1) % (ckp_save_step) == 0):
                print("Saving checkpoint... \n ")
                torch.save(self.state_dict(), ckp_dir + f'{(epoch+1):05d}.pth')



        # print execution time
        print("Total time: %s seconds" % (time.time() - start_time))
        bf.save_object(history_score, score_dir + score_file)

        return history_score
    
    
class inception_resnet_v2_regr(nn.Module):

    def __init__(self, device):

        super(inception_resnet_v2_regr, self).__init__()

        self.device = device
        
        self.inception = timm.create_model('inception_resnet_v2', pretrained=False)


        self.avgPooling = nn.AvgPool2d((9,23))    

        self.linear1 = nn.Linear(1536, 1024)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        
        self.linear3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        
        self.linear4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(16, 1) #steering angle





    def forward(self, x):
        x = self.inception.forward_features(x)
        
        x = self.avgPooling(x).view(x.shape[0], 1536)
        
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.relu3(self.linear3(x))
        x = self.relu4(self.linear4(x))
        
        x = self.linear5(x)

        return x


    def train_model(self, data, max_epoch=40, lr = 1e-3, weight_decay = 0, ckp_save_step=20, log_step=5, ckp_dir = "", score_dir = "", score_file = "score.pkl"):

       # Argument for the training
       #max_epoch          # Total number of epoch
       #ckp_save_step      # Frequency for saving the model
       #log_step           # Frequency for printing the loss



        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        scaler = torch.cuda.amp.GradScaler()
            
        torch.backends.cudnn.benchmark = True
        
        # compute execution time of the cell
        start_time = time.time()



        #Vector to store loss
        history_score = defaultdict(list)

        print("Start Training...\n")


        for epoch in range(max_epoch):

            if (epoch+1) % log_step == 0:
                print("---> Epoch %03i/%03i <--- " % ((epoch+1), max_epoch))

            ###### TRAIN ######
            self.train()

            train_steeringAngle_loss=0
            train_speed_loss = 0
            train_acceleration_loss = 0
            train_tot_loss = 0

            for id_b, batch in tqdm(enumerate(data), total=len(data)):
                # All the gradients are resetted to zero before the training step
                optim.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # We predict target speeds and we save the corresponing GTs
                    pred = self.forward(batch["img"].to(self.device))
                    
    
                    gt_steeringAngle = batch["statistics"][:,0].to(self.device)

    
                    loss = nn.functional.mse_loss(pred.reshape(-1), gt_steeringAngle)

                    train_tot_loss += loss * batch['statistics'].shape[0]

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                


            if (epoch+1) % log_step == 0:
                #print('Total Train Loss: %7.4f - Steering angle Loss: %7.4f - Speed Loss: %7.4f - Acceleration Loss: %7.4f' % (train_tot_loss/len(data), train_steeringAngle_loss/len(data), train_speed_loss/len(data), train_acceleration_loss/len(data)))
                print('Total Train Loss: %7.4f' % (train_tot_loss/len(data)))


            history_score['loss_tot_train'].append((train_tot_loss/len(data)).item())


            # Here we save checkpoints to avoid repeated training
            if ((epoch+1) % (ckp_save_step) == 0):
                print("Saving checkpoint... \n ")
                torch.save(self.state_dict(), ckp_dir + f'{(epoch+1):05d}.pth')



        # print execution time
        print("Total time: %s seconds" % (time.time() - start_time))
        bf.save_object(history_score, score_dir + score_file)

        return history_score