import argparse
import torch
import pickle
from model import CNN_Classifier, DataLoader, Dataset
import sys
import os

def load_data(filename):
    print("Loading data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def get_dataset(pathname):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train.pkl")
    valid_df = load_data(pathname + "valid.pkl")
    test_df = load_data(pathname + "test.pkl")
    return train_df, valid_df, test_df

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The dir path of dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_options()
    hidden_size = 256
    data_path = args.input
    train_df, valid_df, test_df = get_dataset(data_path)
    classifier = CNN_Classifier(result_save_path=data_path.replace("dataset", "models"), epochs=100, hidden_size=hidden_size)
    classifier.preparation(
        X_train=train_df['data'],
        y_train=train_df['label'],
        X_valid=valid_df['data'],
        y_valid=valid_df['label'],
        X_test=test_df['data'],
        y_test=test_df['label'],
    )
    classifier.train()
    
    for metric in ['F1']:
        model_path = os.path.join(data_path.replace("dataset", "models"), f'best_model_{metric}.pt')
        if os.path.exists(model_path):
            print(f"\nLoading the best model for {metric}...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            classifier.model.load_state_dict(torch.load(model_path, map_location=device))

            print(f"\nEvaluating the best model for {metric} on the test set...")
            test_loss, test_score = classifier.test()
            print(f"Test loss: {test_loss}")
            print(f"Test score: {test_score}\n")

        else:
            print(f"\nNo best model found for {metric}.")

if __name__ == "__main__":
    main()