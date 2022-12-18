from libraries import *
import baseFunctions as bf


#========================== Initial preprocess ================================

data = pd.read_csv('./Data/data.csv', sep=';', index_col=0)


data.drop(data.index[:100], inplace=True)

data.drop(data.index[-200:], inplace=True)

def converter(array):
    
    def conv(x):
        x = x.replace(",",".")
        return float(x)


    out = list(map(conv, list(map(str, array))))
    
    return out


data["acceleration"] = converter(data["acceleration"].array)
data["speed"] = converter(data["speed"].array)
data["steeringAngle"] = converter(data["steeringAngle"].array)

data.to_csv("./Data/data.csv")

data["image"] = data["image"].apply(lambda x: "images/"+x)


#========================== Initial preprocess ================================

data2 = pd.read_csv('./Data/data2.csv', index_col=0)


data.drop(data.index[12713:13678], inplace=True)
data.drop(data.index[22194:24044], inplace=True)
data.drop(data.index[30458:30921], inplace=True)


def converter(array):
    
    def conv(x):
        x = x.replace(",",".")
        return float(x)


    out = list(map(conv, list(map(str, array))))
    
    return out


data["acceleration"] = converter(data["acceleration"].array)
data["speed"] = converter(data["speed"].array)
data["steeringAngle"] = converter(data["steeringAngle"].array)

data.to_csv("./Data/data2.csv")


data2["image"] = data2["image"].apply(lambda x: "images2/"+x)



data3 = pd.concat((data, data2), axis=0, ignore_index=True)

data3.to_csv("./Data/imagesTOT.csv")

#=============================== CSV creation =================================


data = pd.read_csv(DATA_ROOT_DIR + 'imagesTOT.csv', index_col=0)


new_data = data.drop(data.loc[(data['steeringAngle'] >-0.5 ) & (data['steeringAngle'] <0.5)].sample(n=100000, random_state=1).index, axis=0)
new_data.drop(new_data.loc[(new_data['steeringAngle'] <-0.5 ) & (new_data['steeringAngle'] >-2)].sample(n=15000, random_state=1).index, axis=0, inplace=True)
new_data.drop(new_data.loc[(new_data['steeringAngle'] >0.5 ) & (new_data['steeringAngle'] <2)].sample(n=15000, random_state=1).index, axis=0, inplace=True)

plt.hist(data["steeringAngle"], bins=80)
plt.hist(new_data["steeringAngle"], bins=80, alpha=0.5)


bf.create_train_test_dataframe(new_data, group_n=1, test_size=0.2, save_dir = "./Data/", test_file_name = "data_test_norm.csv", train_file_name = "data_train_norm.csv", save=True)


data_test = pd.read_csv(DATA_ROOT_DIR + 'data_test_norm.csv', index_col=0)
data_train = pd.read_csv(DATA_ROOT_DIR + 'data_train_norm.csv', index_col=0)

plt.hist(data_test["steeringAngle"], bins=80)
plt.hist(data_train["steeringAngle"], bins=80, alpha=0.5)
