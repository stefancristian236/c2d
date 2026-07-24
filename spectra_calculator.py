import numpy as np
import random


##### macropixel level########

def extract_dataset(data, wl):
    macro_datasets = []
    
    H, W, _ = data.shape
    
    for idx0 in range(H):
        for idx1 in range(W):
            refl = data[idx0, idx1, :]
            refl = refl / np.max(refl)
            macro_datasets.append(
                {
                    'x': wl,
                    'y':refl
                }
            )
    return macro_datasets
       
def extract_sample(sample_size, dataset):
    sample = random.sample(dataset, min(sample_size, len(dataset)))
    return sample

def all_spectra_median(dataset):
    
    y1 = np.array([
        item['y'] for item in dataset
    ])
    
    y = np.median(y1, axis = 0)
    
    return y

def all_spectra_mean(dataset):
    
    y1 = np.array([
        item['y'] for item in dataset
    ])
    
    y = np.mean(y1, axis = 0)
    
    return y