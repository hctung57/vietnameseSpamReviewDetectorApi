import os

labels_task_1 = ["no-spam", "spam"]
labels_task_2 = ["no-spam", "spam-1", "spam-2", "spam-3"]

#directory
DIR_ROOT = './'
STOPWORDS_PATH = os.path.join(DIR_ROOT, 'vietnamese-stopwords-dash.txt')
MODEL_DIR = os.path.join(DIR_ROOT, 'transformer_model')
model_task_1_dir = os.path.join(MODEL_DIR, 'phobert/task_1')
model_task_2_dir = os.path.join(MODEL_DIR, 'phobert/task_2')


#stopword
with open(STOPWORDS_PATH, "r") as ins:
    stopwords = []
    for line in ins:
        dd = line.strip('\n')
        stopwords.append(dd)
    stopwords = set(stopwords)