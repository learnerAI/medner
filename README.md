# Medner

This repository contains code for:
Fine-Tuning BERT (on domain-specific corpora) for masked language modeling.
Training a SpaCy NER Model to recognize specific categories (e.g., micro, macro, prevention, risk factors).
We aim to assess different BERT variants (e.g., Original BERT, BioBERT, ClinicalBERT, and custom BERTs trained on combinations of wiki+Books+data1+data2+data3).
2. Repository Structure
Bert_training_finetuning.ipynb: Jupyter/Colab notebook for fine-tuning the BERT model with MLM.
bert_custom_final.ipynb: Jupyter/Colab notebook for generating NER annotations and training a SpaCy NER model.
config.cfg & base_config.cfg: SpaCy configuration files for training the NER model.
data/: (Folder to be modified) Contains your text data for each model. Replace or rename subfolders (e.g., data 1, data 2, data 3) depending on which dataset you want to train on.
annotations/: Contains corresponding annotation files (e.g., .tsv files) for NER tasks.
3. Setup and Dependencies
Clone or Download the Repository
git clone [YOUR_REPO_URL_HERE]
cd [repo-folder]
Install Dependencies
Make sure you have Python 3.7+ installed.
Install the necessary libraries:

pip install tokenizers transformers datasets spacy-transformers torch
If you’re using Google Colab, many of these are already installed, but you may need to install a few manually (e.g., spacy-transformers).
4. Data Placement and Preparation
Fine-Tuning BERT:
Place your text files in a folder named data 1 (or data 2, data 3, etc.) within a directory named data.
Modify the line:
python
Copy code
text_file_paths = glob.glob("/content/data/data 1/*.txt")to point to the folder you wish to use (e.g., data 2 or data 3) for fine-tuning.
Zip them if needed and unzip in Colab (as shown in the notebook with !unzip "data 1.zip" -d "data").
NER Training:
For NER, ensure that orig_text_dir is set to the folder containing your original text files (e.g., "/data") and annotate_text_dir is set to your annotation folder (e.g., "/annotations").
The code will read each text file in the directory, chunk the text, and use the corresponding annotation file to build entity spans.
5. Steps to Run Fine-Tuning (Model 3, 4, 5, 6, etc.)
Upload/Place Data: For Model 3, upload data 1; for Model 4, upload data 2; for Model 5, data 3; for Model 6, combine them (e.g., data 1 + data 2 + data 3).
Open Bert_training_finetuning.ipynb in Colab (or Jupyter).
Install Libraries (if needed):
!pip install tokenizers transformers datasets
Unzip Data:
!unzip "data 1.zip" -d "data"
Run All Cells:
The notebook will preprocess data, train a WordPiece tokenizer, mask tokens, fine-tune BERT, and finally save your model to a local folder (bert_model_v1).
Change the path (e.g., "data 1" to "data 2") for other models.
6. Steps to Run NER Training
Prepare Annotated Data: Place text files in /data and annotation files in /annotations.
Open bert_custom_final.ipynb in Colab (or Jupyter).
Install Libraries:
!pip install spacy-transformers
Run All Cells in the notebook:
The code will read the text and annotation files, generate a list of (text, {'entities': [...]}), split them into TRAIN, VALID, and TEST sets, and convert them to SpaCy’s .spacy format.
Finally, it will train the NER model and evaluate performance.
Modify config.cfg:
Update train = "/content/train.spacy" and dev = "/content/valid.spacy" if needed, depending on your file system.
Update Transformer Model Path:
Locate the [components.transformer.model] section in config.cfg.
Change the name field to point to your saved or pretrained BERT model. For example, if you have a fine-tuned BERT model saved in bert_model_v1, update as follows:
name = "/content/bert_model_v1"  # Path to your saved BERT model
Alternatively, if using a pretrained model from Hugging Face (e.g., bert-base-uncased), set: name = "bert-base-uncased"
Adjust other hyperparameters in config.cfg as desired.
7. Saving and Using the Trained Models
BERT MLM Model: You can zip and download it from Colab:
!zip -r "/content/drive/MyDrive/bert_model_d1.zip" "bert_model_v1"
NER Model: SpaCy saves the best model in the ./output/model-best directory. Copy or download it to your local machine for future inference or integration into other pipelines.
