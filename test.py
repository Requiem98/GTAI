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




img_name = os.path.join("./Data/images/image4065.jpg")
 
image = io.imread(img_name)

image = image[200:480,:]

image = bf.preprocess(image)

image_n = (image - torch.mean(image)) / torch.std(image)



F.to_pil_image(image)


np.random.randn(3,3).size



mmap = image[520:580,56:116]
mmap[34:43,27:36] = 0
plt.imshow(mmap)


pointer = mmap[35:41,27:34] = 0

plt.imshow(pointer)
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


img_name = os.path.join("./Data/images/image4065.jpg")
 
image = io.imread(img_name)

full_img = image[200:480,:]

image = F.to_pil_image(full_img)
image = F.to_tensor(image)

image = image.unsqueeze(0)


#===============================================================================

segment_data = pd.DataFrame()

segment_data["image"] = glob.glob("images*/*")
segment_data["mask"] = glob.glob("labels*/*")



segment_data.to_csv("segment_data.csv")



ds = bf.GTA_segment_Dataset(csv_file = "segment_data.csv", root_dir="C:/Users/amede/Downloads/segmentation dataset/", transform=bf.preprocess_segment)

out = ds.__getitem__(12)

F.to_pil_image(out["img"])
F.to_pil_image(out["mask"])

out["mask"].shape

img_name = os.path.join("C:/Users/amede/Downloads/segmentation dataset/labels1/00012.png")
 
image = io.imread(img_name)
image = Image.open(img_name)

image = np.array(image)

image = np.array(image)

image2 = np.concatenate(([image], [np.zeros((1052, 1914))], [np.zeros((1052, 1914))]), axis=0)

image2.shape

torch.cat((image.unsqueeze(2), image.unsqueeze(2), image.unsqueeze(2)), dim=2)

image+100

b/2

a/2

plt.imshow(image2.transpose(1,2,0))

image[400:]


F.resize(F.to_pil_image(image[400:900]), (140,400))

plt.imshow(image)


F.to_pil_image(image).load()[0,0]

pred

nn.functional.cross_entropy(torch.rand((1,30,140,400)),  torch.tensor(image.flatten()).unsqueeze(0))

image

np.array(F.resize(image, (140,400)))



nn.CrossEntropyLoss()(torch.rand((1,30,140,400)),  torch.tensor(image[:140, :400], dtype=torch.int64).unsqueeze(0))

mask = np.zeros((1052, 1914), dtype=np.int64)
for i, v in enumerate(range(30)):
    
    mask[image == v] = i

mask.shape
