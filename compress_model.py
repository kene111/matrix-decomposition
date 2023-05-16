import json
import torch as T
import pandas as pd
from tqdm import tqdm


def data_preprocessor():
    pass

print("Load compression configurations ...")
with open('config.json') as f:
    compression_config = f.read()
compression_config = json.loads(compression_config)

print("Load teacher model ...")
teacher_model = T.load(compression_config["teacher_model_path"])
print("Load student model ...")
student_model = T.load(compression_config["student_model_path"])


feature_layers =  compression_config["feature_layers"]
decompose_layers = compression_config["decompose_layers"]

if len(feature_layers) == 0:
    feature_layers = None
if len(decompose_layers) == 0:
    decompose_layers = None

k_info = compression_config["k_info"]

print("Load Decomposition Class ... ")
compress_model = Decomposition(teacher_model, student_model, k_info, feature_layers, decompose_layers)

data = pd.read_csv(compression_config["dataset_path"])
losses = 0

X = data["source_text"].to_list()
Y = data["target_text"].to_list()

print("Compressing Model .. ")
for epoch in tqdm(range(compression_config["epoch"])):
    for i in tqdm(range(compression_config["batch_size"])):
        batch_X, batch_Y = X[i*data_batch : (i+1)*data_batch], Y[i*data_batch : (i+1)*data_batch]
        batch =  data_preprocessor(batch_X, batch_Y)
        losses += compress_model(batch)
    average_loss = losses/data_batch
    print(f'Average Loss Epoch:{epoch}: ' + str(average_loss))

T.save(compress_model.student_model, compression_config["output_path"])
print("End!")
