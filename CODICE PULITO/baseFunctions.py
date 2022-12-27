from libraries import *
from torch import Tensor


#===========================================================================
#=========================== Utils functions ===============================
#===========================================================================



def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def read_object(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def get_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print("Total memory:", t/1000/1000/1000)
    print("Reserved memory:", r/1000/1000/1000)
    print("Allocated memory:", a/1000/1000/1000)
    print("Free memory:", f/1000/1000/1000)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
        



#===========================================================================
#========================= Processing data =================================
#===========================================================================

def create_train_test_dataframe(data, group_n=1, test_size=0.2, save_dir = "./Data/", test_file_name = "data_test.csv", train_file_name = "data_train.csv", save=True):
    
    idx_order = np.array(list(SubsetRandomSampler(list(BatchSampler(SequentialSampler(data.index), batch_size=group_n, drop_last=True)))), dtype=np.int64)
    
    idx_tr, idx_test = train_test_split(idx_order, test_size=test_size)
    
    data_train = data.iloc[idx_tr.flatten()]
    
    data_test = data.iloc[idx_test.flatten()]
    
    if(save):
        data_train.to_csv(save_dir + train_file_name)
        data_test.to_csv(save_dir + test_file_name)
    
    return data_train, data_test





def normalize_steering(x):
    return x /40.5


def reverse_normalized_steering(x):
    return x *40.5


preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((140,400)),
    T.ColorJitter(brightness=(0.5,2)),
    T.RandomAffine(degrees = 0, translate=(0.1,0.1)),
    T.ToTensor()
])


test_preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((140,400)),
    T.ToTensor(),
])


#===========================================================================
#====================== Steering angle dataset =============================
#===========================================================================


class GTADataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, mmap=False, normalize=True, img_dir=""):

        self.statistics = pd.read_csv(root_dir + csv_file, index_col=0)
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.transform = transform
        self.mmap = mmap
        self.normalize=normalize
    

    def __len__(self):
        return len(self.statistics)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        img_names = self.root_dir + self.img_dir + self.statistics.iloc[idx, 3]
    
        if(isinstance(img_names, str)):
            img_names = [img_names]
            
        if(self.mmap):
            mmaps = list()
            
            
        images = list()
        
        for im_name in img_names:
            
            image = io.imread(im_name)
        
            if(self.mmap):
                mmap = image[520:580,56:116]
                mmap = F.to_pil_image(mmap)
                mmap = F.to_tensor(mmap)
                if(self.normalize):
                    mmap = (mmap - torch.mean(mmap)) / torch.std(mmap)
                mmaps.append(mmap)
        
            image = image[200:480, :]
        
            if self.transform:
                image = self.transform(image)
                if(self.normalize):
                    image = (image - torch.mean(image)) / torch.std(image)
        
            images.append(image)
        
        if(len(img_names)>1):
            images = [el.unsqueeze(0) for el in images]
            if(self.mmaps):
                mmaps = [el.unsqueeze(0) for el in mmaps]
            
        images = torch.cat(images)
        
        if(self.mmap):
            mmaps = torch.cat(mmaps)
        
        statistics = self.statistics.iloc[idx, :3]
        statistics = np.array(statistics, dtype=np.float32)
        statistics[0] = normalize_steering(statistics[0])
        statistics = torch.tensor(statistics, dtype=torch.float32)
    

        if(self.mmap):
            sample = {'img': images, 'mmap': mmaps, 'statistics': statistics}
            return sample
            
            
        sample = {'img': images, 'statistics': statistics}

        return sample
    

#===========================================================================
#========================= Steering sampler ================================
#===========================================================================

class SteeringSampler(Sampler):
    def __init__(self, data_source):
        
        super().__init__(data_source=data_source)
        
        self.data = data_source.statistics
        self.weights = np.abs(self.data["steeringAngle"].to_numpy()) + 1e-6
        self.weights = self.weights/np.sum(self.weights)
        self.indexes = np.arange(len(self.data))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        warnings.filterwarnings("error")
        try:
            idx = np.random.choice(self.indexes, p=self.weights, size=1, replace=False)
            self.weights[idx] = 0.0
            self.weights = np.true_divide(self.weights,np.sum(self.weights))
        except:
            idx = np.random.choice(self.indexes, size=1, replace=False)
        warnings.resetwarnings()
      
        return idx[0]
    
    def __len__(self):
        return len(self.data)
    
    def reset_sampler(self):
        self.weights = np.abs(self.data["steeringAngle"].to_numpy()) + 1e-6
        self.weights = self.weights/np.sum(self.weights)