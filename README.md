# BARTViBa

## 1. Training   
- There are multiple models for this project. Use different trainer file and modify configuration for training:
    - **bert2bert**:

    ```> python trainer/custom_bert2bert_trainer.py```
    - **marianmt**:

    ```> python trainer/custom_marianmt_trainer.py```   
    - **bartpho**:

    ```> python trainer/custom_bartpho_trainer.py```   
   
- There is also a colab file at ```trainer_notebook.ipynb``` guiding training process.


## 2. Inference   
* Checkpoint model: 
  The current best model is aligned_bartpho model. the checkpoint is stored at: 
  [Link](https://drive.google.com/drive/folders/10M4l95A7ImxfSPtV8JzAWNm5ytrUwOgC?usp=sharing)
* First: Start VNCoreNLP server at:

    ```> vncorenlp -Xmx2g "D:\AI Learning\NLP\DA\BARTVIBA\GraphTranslation\vncorenlp\VnCoreNLP-1.1.1.jar" -p 9000 -a "wseg,pos,ner,parse"```

    Where ```path_to_VnCoreNLP-1.1.1.jar``` is the full path (or relative path) to the file containing ```VnCoreNLP-1.1.1.jar```

* Then:  Start api server port 8000:

    ```> python app.py```

* 3 model types:
    * **BART**
    * **BART_CHUNK**
    * **BART_CHUNK_NER_ONLY**

## 3. Corpus area
- For each area (Binh Dinh, Gia Lai, Kon Tum), there is a specific corpus data corresponding to that area.

- Data folder:

    * Binh Dinh
    * Gia Lai
    * Kon Tum
- Each folder contains 2 files:

    - Dictionary
    - Parralel corpus
- Dictionary file:

    - dict.ba
    - dict.vi
- Parralel corpus file:

    - train.vi/train.ba
    - valid.vi/valid.ba
    - test.vi/test.ba

- Overall folder structure of ```data```:

```
📦data
 ┣ 📂all
 ┃ ┣ 📜dict.ba
 ┃ ┣ 📜dict.vi
 ┃ ┣ 📜norm_kriem.ba
 ┃ ┣ 📜norm_kriem.vi
 ┃ ┣ 📜test.ba
 ┃ ┣ 📜test.vi
 ┃ ┣ 📜train.ba
 ┃ ┣ 📜train.vi
 ┃ ┣ 📜valid.ba
 ┃ ┗ 📜valid.vi
 ┣ 📂BinhDinh
 ┃ ┣ 📂dictionary
 ┃ ┃ ┣ 📜dict.ba
 ┃ ┃ ┣ 📜dict.vi
 ┃ ┃ ┗ 📜vi-ba_word_dict_norm.json
 ┃ ┗ 📂parallel_corpus
 ┃ ┃ ┣ 📜test.ba
 ┃ ┃ ┣ 📜test.vi
 ┃ ┃ ┣ 📜train.ba
 ┃ ┃ ┣ 📜train.vi
 ┃ ┃ ┣ 📜valid.ba
 ┃ ┃ ┗ 📜valid.vi
 ┣ 📂cache
 ┃ ┗ 📜graph.json
 ┣ 📂GiaLai
 ┃ ┣ 📂dictionary
 ┃ ┃ ┣ 📜dict.ba
 ┃ ┃ ┣ 📜dict.vi
 ┃ ┃ ┗ 📜vi-ba_word_dict_norm.json
 ┃ ┗ 📂parallel_corpus
 ┃ ┃ ┣ 📜test.ba
 ┃ ┃ ┣ 📜test.vi
 ┃ ┃ ┣ 📜train.ba
 ┃ ┃ ┣ 📜train.vi
 ┃ ┃ ┣ 📜valid.ba
 ┃ ┃ ┗ 📜valid.vi
 ┣ 📂KonTum
 ┃ ┣ 📂dictionary
 ┃ ┃ ┣ 📜dict.ba
 ┃ ┃ ┣ 📜dict.vi
 ┃ ┃ ┗ 📜vi-ba_word_dict_norm.json
 ┃ ┗ 📂parallel_corpus
 ┃ ┃ ┣ 📜test.ba
 ┃ ┃ ┣ 📜test.vi
 ┃ ┃ ┣ 📜train.ba
 ┃ ┃ ┣ 📜train.vi
 ┃ ┃ ┣ 📜valid.ba
 ┃ ┃ ┗ 📜valid.vi
 ┣ 📂synonyms
 ┃ ┣ 📜convert.py
 ┃ ┣ 📜vi_syn_data.json
 ┃ ┗ 📜vi_syn_data_1.json
 ┣ 📜newWord.py
 ┗ 📜vi-ba_word_dict_norm.json
```
