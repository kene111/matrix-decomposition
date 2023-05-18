import torch as T
from collections import OrderedDict

class MatrixDecomposition:

  def __init__(self, teacher_layers, k_info=245):
    self.teacher_layers = teacher_layers
    self.k_info = k_info
    self.input_decompose_layers = None

  def select_layers_to_decompose(self, model_layer_names):
    # select the layers of the network to decompose, the layers chosen here are based on the paper.
    layers = []
    for layer in model_layer_names:
      if "encoder" in layer and "weight" in layer:
        if "layer_norm" in layer or "embed" in layer or "decoder" in layer: # these layers are 1-D. Paper talks of matrices with 2-D, I also remove decoder-encoder layers.
          continue
        layers.append(layer)
    return layers

  def decompose(self, A):
    # Decompose Matrix using SVD
    U, S, Vh = T.linalg.svd(A, full_matrices=False)
    return U, S, Vh


  def merge_decomposed_matrix_with_k_info(self,U,S,V,k):
    # Merge decomposed matrix using k amount of information
    S = T.diag(S)
    merged_decomposed_matrix = U[:, :k] @ S[0:k, :k] @ V[:k, :]
    return merged_decomposed_matrix

    
  def get_and_decompose_teacher_layers_into_student(self, student_layers):
    # get the specific layers from the teacher model or user input
    layers_to_decompose = None
    if self.input_decompose_layers is not None:
      layers_to_decompose = self.select_layers_to_decompose(self.input_decompose_layers)
    else:
      layers_to_decompose = self.select_layers_to_decompose(self.teacher_layers.keys())

    # decompose the teacher model layers into the student layer
    for layer in layers_to_decompose:
      Wm = self.teacher_layers[layer]
      U, S, VT = self.decompose(Wm)
      k_decomposed_matrix = self.merge_decomposed_matrix_with_k_info(U,S,VT, self.k_info)
      student_layers[layer] = k_decomposed_matrix 
    return student_layers


  def initialize_student_layers(self, student_model, decompose_layers=None):
    # initialize and return the decomposed student model 
    student_model_layer_info = student_model.state_dict()
    if decompose_layers is not None:
      self.input_decompose_layers = decompose_layers

    decomposed_layers = self.get_and_decompose_teacher_layers_into_student(student_model_layer_info)
    student_model.load_state_dict(decomposed_layers)
    return student_model
