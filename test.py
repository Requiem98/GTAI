from libraries import *
import baseFunctions as bf
from models import MapNet
CKP_DIR = "./Data/models/Inception/checkpoint/"



data = pd.read_csv(DATA_ROOT_DIR + 'data.csv', index_col=0)


DATA_ROOT_DIR + "images/" + data.iloc[[1,2,3,4], 3]


#Check loss history
scores = bf.read_object("./Data/models/CNN/scores/00085_history_score.pkl")

plt.plot(scores['loss_tot_train'][1:])

plt.plot(scores['MAE_train'][1:])

scores["MAE_train"]



# Load pre-trained backbone
inception = timm.create_model('inception_resnet_v2', pretrained=False)

state_dict_backbone = bf.get_backbone_state_dict(torch.load(CKP_DIR + f'{(21):05d}.pth'), "inception")

inception.load_state_dict(state_dict_backbone, strict=False)




img_name = os.path.join("./Data/images/image2000.jpg")
 
image = io.imread(img_name)

image = image[200:480,:]

image = bf.preprocess(image)

F.to_pil_image(image)

torch.mean(image)





mmap = image[520:580,56:116]
plt.imshow(mmap)
mmap.shape

plt.imshow(image)

image.shape

image2 = bf.preprocess(image)

plt.imshow(image2.permute(1,2,0))


        
train_dataset = bf.GTADataset("data_train_norm.csv", DATA_ROOT_DIR, bf.preprocess, mmap=True)

train_dl = DataLoader(train_dataset, 
                        batch_size=2,
                        sampler=bf.SteeringSampler(train_dataset), 
                        num_workers=0)


for b in tqdm(train_dl, total=len(train_dl)):
    break



mapnet = MapNet(device).to(device)

mapnet(b["img"].to(device), b["mmap"].to(device))




sct = mss.mss()
mon = {'top': top, 'left': left, 'width': 800, 'height': 600}

sct_img = sct.grab(mon)
image = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
image








