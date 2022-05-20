# CS60075_Course_Project_Multi_CoNER


Group members:
- Kaizer Rahaman (19IE10044)
- SREEYA         (19AG10008)


## Code Repository Organisation
```bash
├── Data/
|    ├── BN-Bangla
|    ├── EN-English
|    └── HI-Hindi
|    └── bn_dev_features.csv
|    └── bn_train_features.csv
|    └── en_dev_features.csv
|    └── en_train_features.csv
|    └── hn_dev_features.csv
|    └── hn_train_features.csv
├── Code With Outputs/
|    ├── Additional Features Implementation Code.ipynb
|    ├── Baseline Code.ipynb
|    └── Fine Tuning Dense Layers Code.ipynb
|    └── Multilingual-Transformers+BiLSTM-CRF Code.ipynb
|    |__ MonoLingual - Transformers+BiLSTM-CRF Code.ipynb
├── src/
|    ├── data_reader.py
|    ├── global_var.py
|    └── main.py
|    └── metric.py
|    └── models.py
|    └── utils.py

├── README.md
```



```

All the code and Data will get downloaded by the above command.
In order to reproduce our results,
- Files run on Google Colab, hence directly running on Colab will be beneficial for quick checking 
- Open any choice of the IPython Notebook present in the 'Code with Outputs' folder
- In order to change the parameters for your running the notebook
- Just change the parameter settings in **Args Class** in the IPython Notebook and then do run all (for Multilingual-Transformers+BiLSTM-CRF Code where all test file paths are to be explicitly mentioned)
- Provide correct path to the dataset and features.csv files (for English,Hindi and Bangla) for the language for which you are running the code (especially for Multilingual-Transformers+BiLSTM-CRF Code where all test file paths are to be explicitly mentioned)
- The model will be saved after all the code cells get executed.

