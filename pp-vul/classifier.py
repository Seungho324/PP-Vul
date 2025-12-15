import argparse
import torch
import pickle
from he_friendly_model import CNN_Classifier
import sys

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
    
    classifier = CNN_Classifier(epochs=100, hidden_size = hidden_size)
    classifier.preparation(
        X_train=train_df['data'],
        y_train=train_df['label'],
        X_valid=valid_df['data'],
        y_valid=valid_df['label'],
        X_test=test_df['data'],
        y_test=test_df['label'],
    )
    classifier.train()

    print("\nLoading best F1 model for testing...")
    classifier.model = torch.load('./pp-vul_16_4_x^3.pth', map_location=classifier.device)
    classifier.model.to(classifier.device)

    print(f"\nEvaluating the best model on the test set...")
    test_loss, test_score = classifier.test()       
    print(f"Test loss: {test_loss}")
    print(f"Test score: {test_score}\n")

if __name__ == "__main__":
    main()