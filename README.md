# Natural-Language-Processing/Eng-Hin-NMT

An implementation of English to Hindi Neural Machine Translation. This model is trained on 140,000 sentences taken from IIT Bombay English Hindi Parallel Corpus.


A preview of model configuration

1) Embedding Layer

2) Bi-directional LSTM #Encoder.

3) Repeat Vector #for connecting encoder to decoder as input and output shapes might be different.

4) Bi-directional LSTM #Decoder

5) Dense #to get output in desired shape.


Dataset Used:

IIT Bombay English-Hindi parallel corpus. The dataset can be downloaded from http://www.cfilt.iitb.ac.in/iitb_parallel/

Steps to run:

1)   Follow steps mentioned in preprocess.py for cleaning and loading of dataset.

2) 1) You can run this notebook in Google Colab by just clicking on the link mentioned right at the top of the notebook.
   2) If you want to run it on local machine (not recommended as it needs GPU) you can install dependencies via        requirements.txt then preprocess.py then the main nmt file.
