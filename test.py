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


img_name = os.path.join("./Data/images/image200.jpg")
 
image = io.imread(img_name)

image = image[:480,:]

plt.imshow(image)

image2 = bf.preprocess(image)

plt.imshow(image2.permute(1,2,0))




#create csv files
data = pd.read_csv(DATA_ROOT_DIR + 'data.csv', index_col=0)


new_data = data.drop(data.loc[(data['steeringAngle'] >-0.5 ) & (data['steeringAngle'] <0.5)].sample(n=100000, random_state=1).index, axis=0)
new_data.drop(new_data.loc[(new_data['steeringAngle'] <-0.5 ) & (new_data['steeringAngle'] >-2)].sample(n=15000, random_state=1).index, axis=0, inplace=True)
new_data.drop(new_data.loc[(new_data['steeringAngle'] >0.5 ) & (new_data['steeringAngle'] <2)].sample(n=15000, random_state=1).index, axis=0, inplace=True)

plt.hist(data["steeringAngle"], bins=80)
plt.hist(new_data["steeringAngle"], bins=80, alpha=0.5)


new_data

#bf.create_train_test_dataframe(new_data, group_n=1, test_size=0.2, save_dir = "./Data/", test_file_name = "data_test_norm.csv", train_file_name = "data_train_norm.csv", save=True)









