from pp_vul import *
from he_friendly_model import CNN_Classifier, CNN, Dataset
from pp_vul.utils.structure import *

from seal import *
from torchvision import datasets
import numpy as np
import torch
from tqdm import tqdm
import sys, os
import argparse
import pickle

MAX_LEN = 64
HIDDEN_SIZE = 256

def load_data(filename):
    print("Loading data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The dir path of dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_path = parse_options().input
    data_path = data_path + "/" if data_path[-1] != "/" else data_path
    test_df = load_data(data_path + "test.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN(hidden_size = HIDDEN_SIZE)
    model = torch.load(root_dir + '/pp-vul_16_4_x^3.pth', map_location=device)
    
    context = Context(N = 2**15, depth = 5, LogQ = 40, LogP = 60)
    embedding_size = Cuboid(1, MAX_LEN, HIDDEN_SIZE)

    HE_vul = HE_CNN(model, embedding_size, context)
    print(HE_vul)
    print('='*50)

    num_of_data = 1

    X_test, y_test = test_df['data'], test_df['label']
   
    num_of_test = 10
    
    for k in range(num_of_test):
        data = np.zeros(shape=(1, MAX_LEN, HIDDEN_SIZE))
        for j in range(1):
            for i in range(min(len(X_test[k]) - 1, MAX_LEN - 1)):
                data[j][i + 1] = X_test[k][i + 1]

        ppData = preprocessing(data, embedding_size, num_of_data, HE_vul.data_size)
        
        ciphertext_list = HE_vul.encrypt(ppData)
        
        result = model(torch.tensor(data, dtype=torch.float32).to(device)).tolist()
        
        print('='*50)
    
        result_ciphertext = HE_vul(ciphertext_list, _time=True)

        result_decrypted = HE_vul.decrypt(result_ciphertext)[:2]

        print("Plaintext result:", result)
        print("Ciphertext result:", result_decrypted)
        
        result_1 = 3
        result_2 = 3
        if (result[0] > result[1]):
            result_1 = 0
        else:
            result_1 = 1
        if (result_decrypted[0] > result_decrypted[1]):
            result_2 = 0
        else:
            result_2 = 1
        print("Plaintext result:", result_1)
        print("Ciphertext result:", result_2)
        
        if result_1 == result_2:
            print("True")
        else:
            print("False")
    

if __name__ == "__main__":
    main()