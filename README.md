# PP-Vul: Privacy-Preserving Vulnerability Detection Using Homomorphic Encryption

This repository contains the implementation of *PP-Vul*.

## Requirements
We have confirned that PP-Vul can be executed on a 64-bit Ubuntu 18.04 system with python

- Python 3.10
- SEAL-Python 4.0 (https://github.com/Huelse/SEAL-Python)
- PyTorch

## How to use PP-Vul

### 1. Dataset pre-processing
- The raw dataset of VulCNN can be found at: https://github.com/CGCL-codes/VulCNN/tree/main/dataset
- The raw dataset of BigVul can be found at: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
- The raw dataset of PrimeVul can be found at: https://github.com/DLVulDet/PrimeVul
- Normalization code can be found at: https://github.com/CGCL-codes/VulCNN/blob/main/normalization.py
- For convenience, the preprocessed dataset of VulCNN can be found at: https://drive.usercontent.google.com/download?id=1NGslBtKLA1gUex4tjNWbF01O3sFLSHPD&export=download&authuser=1
- If you are using the preprocessed dataset, *Dataset pre-precessing* can be skipped.
  
#### 1.1. Normalization

- Download the raw dataset from the link (https://github.com/CGCL-codes/VulCNN/tree/main/dataset) into *dataset* folder. 
- Unzip the dataset and rename the folder

```python
mv Dataset-sard rawdata
```

- Normalize the dataset
```python
python normalization.py -i ../dataset/rawdata
```

#### 1.2. Embedding    
- Move to *preprocessing* folder.

```python
python codet5embedding.py -i ../dataset/normalized/Vul -o ../dataset/embedding/1_line/Vul -n 1
python codet5embedding.py -i ../dataset/normalized/No-Vul -o ../dataset/embedding/1_line/No-Vul -n 1
```
#### 1.3. Split dataset (train:valid:test=7:2:1)

```python
python split_data.py -i ../dataset/embedding -o ../dataset/1_line 
```
### 2. PP-Vul

#### 2.1. Model training (HE-friendly)

- Default model: Kernel height = 4, Number of filters = 16, Activation function = $x^3$, Depthwise convolution
- Move to *pp-vul* folder.

```python
python classifier.py -i ../dataset/1_line 
```

#### 2.2. Privacy-preserving inference using homomorphic encrytion

```python
python main.py -i ../dataset/1_line 
```

### 3. Plaintext model

- Default model: Number of kernel = 10, Number of filters = 32, Activation function = ReLU, Max pooling
- Move to *plaintext_model* folder.

```python
python classifier.py -i ../dataset/1_line
```

