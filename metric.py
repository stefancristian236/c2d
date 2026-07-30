import numpy as np

def rmse(array : np.array) -> float:
    
    n_elem = np.size(array)
    return np.sqrt(np.sum(np.square(array - np.mean(array))) / n_elem)

def macropixel_error(arr : np.array) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    
    err = np.zeros((arr.shape[0], arr.shape[1]))
    
    for idx0 in range(arr.shape[0]):
        for idx1 in range(arr.shape[1]):
            err[idx0, idx1] = rmse(arr[idx0,idx1, :])
    
    aux = err.flatten()
    
    ermax = round(np.max(aux) * 100, 4)
    ermin = round(np.min(aux) * 100, 4)
    ermed = round(np.median(aux) * 100, 4)
    
    return ermax, ermin, ermed, err