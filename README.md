# CLSESSP Main Code

## requirements
```bash
pip install requirements.txt
```

If you download the latest version of the transformer library, please use the following command to install the required version of the tokenizers library:  

```bash
pip install tokenizers==0.9.4
```

## Model
You should download the pre-trained models from Hugging Face and save them in a path like: model/bert-base-uncased/... 

## Data link

You can download the data files from this link：https://drive.google.com/file/d/1DinUy1eao7P_59aZGp0BqMRd20ZR5Qb7/view?usp=drive_link

### wiki1m_for_simcse.txt

This file is used for unsupervised training. You should create a directory and path like this:

```bash
data/wiki1m_for_simcse.txt
```

### data_for_senteval

These files are used for the SentEval library. You should create a directory and path like this:

```bash
SentEval/data/...
```
