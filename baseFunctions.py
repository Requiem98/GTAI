from libraries import *

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


def torch_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


class SteeringSampler():
    def __init__(self, path_to_csv):
        
        self.data = pd.read_csv(path_to_csv, index_col=0)
        self.weights = np.abs(self.data["steeringAngle"].to_numpy()) + 1e-6
        self.weights = self.weights/np.sum(self.weights)
        self.indexes = np.arange(len(self.data))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            idx = np.random.choice(self.indexes, p=self.weights, size=1, replace=False)
            self.weights[idx] = 0.0
            self.weights = np.true_divide(self.weights,np.sum(self.weights))
        except:
            idx = np.random.choice(self.indexes, size=1, replace=False)
      
        return idx[0]
    
    def __len__(self):
        return len(self.data)
    
    def reset_sampler(self):
        self.weights = np.abs(self.data["steeringAngle"].to_numpy()) + 1e-6
        self.weights = self.weights/np.sum(self.weights)
        


def normalize_steering(x):
    x = x+40.5
    return np.true_divide(x, 80.5)


def reverse_normalized_steering(x):
    x = x*80.5
    return x - 40.5
    


preprocess = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])


class GTADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, load_all=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.statistics = pd.read_csv(root_dir + csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
    
                

    def __len__(self):
        return len(self.statistics)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
       

        img_name = os.path.join(self.root_dir + "images/",
                                self.statistics.iloc[idx, 3])
        
        image = io.imread(img_name)
        
        image = image[:480, :]
        
        if self.transform:
            image = self.transform(image)
    
        statistics = self.statistics.iloc[idx, :3]
        statistics = np.array(statistics, dtype=np.float32)
        statistics[0] = normalize_steering(statistics[0])
        statistics = torch.tensor(statistics, dtype=torch.float32)
    

            
            
        sample = {'img': image, 'statistics': statistics}

        return sample