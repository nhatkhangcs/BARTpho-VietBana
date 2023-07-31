# BARTViBa

# HOW TO

1. Training   
There are multiple models for this project. Use different trainer file and modify configuration for training:
    * bert2bert: 
    >python trainer/custom_bert2bert_trainer.py   
    * marianmt: 
    >python trainer/custom_marianmt_trainer.py   
    * bartpho:
    >python trainer/custom_bartpho_trainer.py   
   
There is also a colab file at <trainer_notebook.ipynb> guiding training process.


2. Inference   
* Checkpoint model: 
  The current best model is aligned_bartpho model. the checkpoint is stored at: https://drive.google.com/drive/folders/10M4l95A7ImxfSPtV8JzAWNm5ytrUwOgC?usp=sharing  
* First: Start VNCoreNLP server at:
    >vncorenlp -Xmx2g GraphTranslation/vncorenlp/VnCoreNLP-1.1.1.jar -p 9000 -a "wseg,pos,ner,parse"
* Then:  Start api server port 8000:
    >python app.py
* 3 model types:
    * BART
    * BART_CHUNK
    * BART_CHUNK_NER_ONLY