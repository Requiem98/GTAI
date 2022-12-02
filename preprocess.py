from libraries import *
import baseFunctions as bf

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




