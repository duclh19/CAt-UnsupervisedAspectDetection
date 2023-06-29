__[IT4772E-NLP]__

Capstone project for Natural Language Processing course, SoICT, HUST, Spring 2023. 

Implementation of CAt approach for Unsupervised Aspect Category Detection. The original method is proposed in the paper: "Embarrassingly Simple Unsupervised Aspect Extraction" by Tulkens and Cranenburgh, 2020. 


## 1. Settings

  - 1.1 Download the dataset: 
    
    The dataset can be downloaded by [SemEval-2014]() and [CitySearch](). 
    
    Put the SemEval-2014 dataset in `data/semeval2014/` folder, and CitySearch dataset in `data/citysearch/` folder.

  - 1.2 Preprocessing. Run: 

    `python embeddings/preprocessing.py`
## 2. Experiments


  - Experiment on CitySearch dataset or SemEal-2014 restaurant dataset.
    
    `python main.py`


  - Test your own experiment in restaurant domain.
  
    `python inference.py`

