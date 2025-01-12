import torch
import pandas as pd
import os

def load_csv_to_tensor(file_path, expected_shape=None):
    data = pd.read_csv(file_path, header=None, sep=";")

    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    tensor = torch.tensor(data.values, dtype=torch.float32)
    
    if expected_shape and tensor.shape != expected_shape:
        raise ValueError(f"Arquivo {file_path} tem forma {tensor.shape}, mas esperava {expected_shape}")
    
    return tensor

def save_tensor_to_csv(tensor, file_path):

    df = pd.DataFrame(tensor.numpy())
    df.to_csv(file_path, index=False, header=False, sep=";")

def compare_tensors(tensor1, tensor2, atol=1e-6):
    return torch.allclose(tensor1, tensor2, atol=atol)

def main():
    data_folder = "./data"
    files = {
        "a": ("a.csv", (1, 10)),
        "M": ("M.csv", (10, 10)),
        "N": ("N.csv", (10, 10)),
        "aM": ("aM.csv", (1, 10)),
        "MN": ("MN.csv", (10, 10)),
    }
    
    tensors = {}
    for key, (file, shape) in files.items():
        tensors[key] = load_csv_to_tensor(os.path.join(data_folder, file), expected_shape=shape)
    
    MN = torch.mm(tensors["M"], tensors["N"])
    aM = torch.mm(tensors["a"], tensors["M"])
    Ma = torch.mm(tensors["M"], tensors["a"].T)

    # MN
    if compare_tensors(MN, tensors["MN"]):
        print("MN: Os resultados são consistentes com o arquivo de referência.")
    else:
        print("MN: Os resultados são diferentes do arquivo de referência.")
    save_tensor_to_csv(MN, os.path.join(data_folder, "MN_calculated.csv"))

    # aM
    if compare_tensors(aM, tensors["aM"]):
        print("aM: Os resultados são consistentes com o arquivo de referência.")
    else:
        print("aM: Os resultados são diferentes do arquivo de referência.")
    save_tensor_to_csv(aM, os.path.join(data_folder, "aM_calculated.csv"))

    # Ma
    save_tensor_to_csv(Ma, os.path.join(data_folder, "Ma_calculated.csv"))
    print("Ma: Resultado salvo em Ma_calculated.csv.")

if __name__ == '__main__':
    main()
