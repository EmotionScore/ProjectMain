

# Reproducing results of research "Emotion and reason in political language" & replication on new dataset

Welcome! This file provides overview on the project, as well as detailed instructions on running code present in the repository. Make sure you eead carefully, before reproducing results.

## Aim of the project
This project was inspired by article “Emotion and Reason in Political Language” (2022) by Gennaro, G., & Ash, E. Authors of above research are sharing all data & codes used in research with public: https://zenodo.org/records/5748084   
Project consists of two parts:   
1. in the first one, authors reproduce results of abovementioned article - the figures used in the final article. Codes are translated from python to R.
2. In the second part, authors set up and run the replication pipeline, inspired by “Emotion and Reason in Political Language", but using other dataset.

---

## Instructions
### 1. Reproducing figures from article

1. Download replication package from: https://drive.google.com/file/d/1Hv9sS-WN6Dnjclfu_bRbp5Mnsodu_wvw/view?usp=sharing  
2. Unzip replication package to convenient location. Inside the extracted folder, you will find:
- `data` – Contains all dataset for main project
- `models` – models which created by authors
- `scripts/`, `results/` – Supporting files by author  
3. R codes for figures replication are available in R_version folder. Open the scripts and adjust working directory as well as paths to data files  
   Important! Keep the original structure of files from replication package to avoidadditional steps
4. Open and run the main script:
   - `fig1.R` or `fig2.R`
This will generate figures that replicate the original Python results.

### 2. Replicating pipeline on other dataset  

1) Dowload the replication_package folder to your dekstop
2) Download dataset and add files to folder "data" inside replciation package. Access the dataset from the following Google Drive link:
https://drive.google.com/drive/folders/1WAEHlOCzH-92BbyZEeYKXs8HuzjORlL7
 
3) Then go scripts folder inside create "1_document_data_pre_processing" folder and 4 py files there, following the numerical order (1_extract_articles, 2_indexed speeches, 3_word_frequencies, 4_final_article_cleaning.py)

4) Also dowload relative python packages and libraris befor running. Be mindful about packages compatibility. To ensure code runs correctly, we recommend using necessary packages in versions:  
numpy	1.24.4  
scipy	1.10.1  
spacy	3.5.3  
thinc	8.1.10  
h5py	3.9.0  
nltk	3.8.1  

5) Continue running scripts from folder 2_model training and 3_create_dictionaries, following the numerical order.. The outputs of the scripts will be stored in folders: "data", "models" and "results"





