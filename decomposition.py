from torch import nn
from matrix_decomposition import MatrixDecomposition
from distillation.distillation_loss import DistillationLoss
from distillation.knowledge_distillation import KnowledgeDistillation


class Decomposition(nn.Module):

  def __init__(self,teacher, student, k_info, feature_layers, decompose_layers):
    super().__init__()
    self.teacher = teacher
    model_layers_info = self.teacher.state_dict()
    self.matrix_decomposition = MatrixDecomposition(model_layers_info, k_info=k_info)
    # Initialize student model encoder weights with knowledge decompostion
    self.student = self.matrix_decomposition.initialize_student_layers(student, decompose_layers)
    self.loss = DistillationLoss()
    if feature_layers is None:
      feature_layers = self._get_feature_layers(model_layers_info)
    self.knowledge_distillation = KnowledgeDistillation(self.teacher,self.student, self.loss, feature_layers)

  def _check_exclude_layers(self, layer, exclude_layers):
    for exclude in exclude_layers:
      if exclude in layer:
        return True
    return False


  def _get_feature_layers(self, model_layers):
    # get the layers in the next work excluding the layers in the exclude_layers list
    feature_layers = []
    exclude_layers = ["layer_norm","embed","lm_head","shared.weight", "final_logits_bias"]
    for layer in model_layers:
      if self._check_exclude_layers(layer, exclude_layers):
        continue
      feature_layers.append(layer)
    return feature_layers
      
   
  def forward(self, batch):
    return self.knowledge_distillation(batch)