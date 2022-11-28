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
        
        image = image[:350, :]
        
        if self.transform:
            image = self.transform(image)
    
        statistics = self.statistics.iloc[idx, :3]
        statistics = np.array(statistics, dtype=np.float32)
        statistics = torch.tensor(statistics, dtype=torch.float32)

            
            
        sample = {'img': image, 'statistics': statistics}

        return sample