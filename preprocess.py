from libraries import *
import baseFunctions as bf


#========================== Initial preprocess ================================

data = pd.read_csv('./Data/data3.csv', sep=';', index_col=0)


idx = np.concatenate([data.index[:126], data.index[745:]])

data.drop(idx, inplace=True)



def converter(array):
    
    def conv(x):
        x = x.replace(",",".")
        return float(x)


    out = list(map(conv, list(map(str, array))))
    
    return out


data["acceleration"] = converter(data["acceleration"].array)
data["speed"] = converter(data["speed"].array)
data["steeringAngle"] = converter(data["steeringAngle"].array)


data["steeringAngle"].hist(bins=80)


data["image"] = data["image"].apply(lambda x: "images2/"+x)

data.to_csv("./Data/images2.csv")


#========================== Initial preprocess ================================

images = pd.read_csv('./Data/images.csv', index_col=0)
images2 = pd.read_csv('./Data/images2.csv', index_col=0)
images4 = pd.read_csv('./Data/images4.csv', index_col=0)
images5 = pd.read_csv('./Data/images5.csv', index_col=0)
images6 = pd.read_csv('./Data/images6.csv', index_col=0)
images7 = pd.read_csv('./Data/images7.csv', index_col=0)
images8 = pd.read_csv('./Data/images8.csv', index_col=0)
images9 = pd.read_csv('./Data/images9.csv', index_col=0)
images10 = pd.read_csv('./Data/images10.csv', index_col=0)


data_tot = pd.concat((images, images2, images4, images5, images6, images7, images8, images9, images10), axis=0, ignore_index=True)

data_tot.to_csv("./Data/imagesTOT.csv")

#=============================== CSV creation =================================


data_tot = pd.read_csv(DATA_ROOT_DIR + 'imagesTOT.csv', index_col=0)


plt.hist(data_tot["steeringAngle"], bins=80)
plt.hlines(4000, -40, 40)
plt.xticks(np.arange(-40,45, 5))



values, bins = np.histogram(data_tot["steeringAngle"], bins=80)


new_data = data_tot.copy()

for i, count in enumerate(values):
    if(count >15000):
        new_data.drop(new_data.loc[(new_data['steeringAngle'] >bins[i] ) & (new_data['steeringAngle'] <bins[i+1])].sample(n=count-15000, random_state=1).index, axis=0, inplace=True)


plt.hist(new_data["steeringAngle"], bins=80)

new_data.to_csv("./Data/imagesTOT_balanced.csv")

bf.create_train_test_dataframe(new_data, group_n=1, test_size=0.2, save_dir = "./Data/", test_file_name = "data_test.csv", train_file_name = "data_train.csv", save=True)


data_test = pd.read_csv(DATA_ROOT_DIR + 'data_test.csv', index_col=0)
data_train = pd.read_csv(DATA_ROOT_DIR + 'data_train.csv', index_col=0)

plt.hist(data_test["steeringAngle"], bins=80)
plt.hist(data_train["steeringAngle"], bins=80, alpha=0.5)


#=============================== Segmentation dataset ============================


data = pd.read_csv("C:/Users/amede/Downloads/segmentation dataset/segment_data.csv", index_col=0)

data_train, data_test = bf.create_train_test_dataframe(data, group_n=1, 
                                                       test_size=0.2, 
                                                       save_dir = "C:/Users/amede/Downloads/segmentation dataset/", 
                                                       test_file_name = "segment_data_test.csv", train_file_name = "segment_data_train.csv", save=True)

