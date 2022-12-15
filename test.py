from libraries import *
import baseFunctions as bf
CKP_DIR = "./Data/models/Inception/checkpoint/"



data = pd.read_csv(DATA_ROOT_DIR + 'data.csv', index_col=0)


DATA_ROOT_DIR + "images/" + data.iloc[[1,2,3,4], 3]


#Check loss history
scores = bf.read_object("./Data/models/CNN/scores/00020_history_score.pkl")

plt.plot(scores['loss_tot_train'][15:])

plt.plot(scores['MAE_train'][15:])




# Load pre-trained backbone
inception = timm.create_model('inception_resnet_v2', pretrained=False)

state_dict_backbone = bf.get_backbone_state_dict(torch.load(CKP_DIR + f'{(21):05d}.pth'), "inception")

inception.load_state_dict(state_dict_backbone, strict=False)


img_name = os.path.join("./Data/images/image2000.jpg")
 
image = io.imread(img_name)

image = image[:480,:]

mmap = image[480:580,10:160]

plt.imshow(mmap)


plt.imshow(image)

image2 = bf.preprocess(image)

plt.imshow(image2.permute(1,2,0))






