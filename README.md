# matrix-decomposition
This repo contains the implementation of the matrix decomposition compression technique from the paper ["Compressing Pre-trained Language Models by Matrix Decomposition"](https://aclanthology.org/2020.aacl-main.88/) using the pytorch framework.


### Repository Breakdown:
  1. distillation: This folder contains the ```distillation_loss.py``` and the ```knowledge_distillation.py``` module.
  2. matrix_decomposition.py: This python script contains a class object used to decompose matrices using SVD.
  3. config.json: This json file contains the necessary parameters for the application.
  4. compress_model.py: This script runs the application.
  5. decomposition.py: This scripts brings together  ```distillation``` module and ``` matrix_decomposition.py```  as implemented in the paper.
  
 #### NOTE: The paper implements the process for a ["BERTModel"](https://huggingface.co/blog/bert-101), this implemention was done for a ["MarianMTmodel"](https://huggingface.co/docs/transformers/model_doc/marian#transformers.MarianMTModel).

### To Know:
1. The intput into the MarianMTModel is an ```OrderedDict```, with keys ```"input_ids", "attention_mask", and "labels"```.
2. Update the ```data_preprocessor``` in ```compress_model.py``` file to handle how to preprocess your data.
3. Specify the layers you want to decompose and the layers you want to perform feature distillation on by passing in the actual key strings of the model into the lists in the config.json
 e.g : 
 ``` 
        { 
        ...,
        "feature_layers": ['model.encoder.layers.0.self_attn.k_proj.weight', 'model.encoder.layers.0.self_attn.k_proj.bias', ...],
        "decompose_layers": ['model.encoder.layers.0.self_attn.k_proj.weight', ...],
        ... 
        }
  ```
 4. For your data, make sure the data column names are ```source_text``` and ```target_text```.
 5. k_info is the same as rank r used in the paper. 
    ```The dimension of a matrix A is n x d, the decomposed matrix is A = B' * C' and their dimensions are n x r and r x d respectively. Where r < nd/n+d. Think about it as how much information should be retained in the decomposed matrix.```
 
### To run application:
1. Update the config.json file with the required parameters
2. Run the following:
```python compress_model.py```

