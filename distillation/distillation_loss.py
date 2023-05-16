import torch as T
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

class DistillationLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.crossentropy_loss = nn.CrossEntropyLoss(ignore_index=64171)
    self.distillation_loss = nn.KLDivLoss()
    self.softmax = nn.Softmax(dim=-1)
    self.alpha =  0.7**5
    self.temperature = 10

  def _order_qkv_projections(self, layers):
    proj_dict = defaultdict(list)

    for vals in layers:
      if "bias" in vals:
        continue
      if "q_proj" in vals:
        proj_dict["q"].append(vals)
      if "k_proj" in vals:
        proj_dict["k"].append(vals)
      if "v_proj" in vals:
        proj_dict["v"].append(vals)

    return proj_dict

  def _calc_attn_mtx(self, layer_state_dicts, q_projections, k_projections):
      Amtx_list = []
      for q,k in zip(q_projections,k_projections):    
        q = layer_state_dicts[q]
        k = layer_state_dicts[k]
        kt = k.transpose(0,1)
        Qkt = T.matmul(q,kt)
        sftmx_ = self.softmax(Qkt)
        Amtx_list.append(sftmx_)
      return Amtx_list  

  def _calc_attn_matrix_loss(self, teacher, student, q_projections, k_projections):

      losses  = 0

      teacher_attn_matrix = self._calc_attn_mtx(teacher, q_projections, k_projections)
      student_attn_matrix = self._calc_attn_mtx(student, q_projections, k_projections)

      assert len(teacher_attn_matrix) == len(student_attn_matrix),"teacher and student batch attn matix length is not the same."
      len_ = len(teacher_attn_matrix)

      for i in range(len_ ):
        loss = T.dist(teacher_attn_matrix[i], student_attn_matrix[i], p=2).item()
        losses += loss
      return losses / len_ , teacher_attn_matrix, student_attn_matrix

  def _calc_attn_head(self, qk_batch, state_dict_layer, v_batch):

      batch_qkT_v = []
      for qk, v in zip(qk_batch, v_batch):
        v = state_dict_layer[v]
        qkT_v = T.matmul(qk, v)
        batch_qkT_v.append(qkT_v)

      return batch_qkT_v

  def _calc_attn_head_loss(self, teacher_attn_matrix, student_attn_matrix, teacher, student, v):
      losses = 0
      student_attn_head = self._calc_attn_head(student_attn_matrix, student, v)
      teacher_attn_head = self._calc_attn_head(teacher_attn_matrix, teacher, v)

      assert len(teacher_attn_head) == len(student_attn_head),"teacher and student batch attn head length is not the same."
      len_ = len(teacher_attn_head)

      for i in range(len_):
        loss = T.dist(teacher_attn_head[i], student_attn_head[i], p=2).item()
        losses += loss
      return losses/len_
      
  def get_crossentropy_loss(self, y_, y):
    cross_entropy_output_loss = self.crossentropy_loss(y_, y)
    return cross_entropy_output_loss
    
  def get_kd_loss(self, teacher_model_logits, student_model_logits):
    distill_loss = self.distillation_loss(F.log_softmax(student_model_logits / self.temperature, dim=1),
                                                   F.softmax(teacher_model_logits / self.temperature, dim=1))
    return distill_loss

  def get_fd_loss(self, teacher, student, layers):

    loss_ = 0
    qkv_proj = self._order_qkv_projections(layers)

    q_projection_tags = qkv_proj["q"]
    k_projection_tags = qkv_proj["k"]
    v_projection_tags = qkv_proj["v"]
    
    batch_attn_matrix_loss, batch_teacher_attn_matrix, batch_student_attn_matrix = self._calc_attn_matrix_loss(teacher, student, q_projection_tags, k_projection_tags)
    batch_attn_head_loss = self._calc_attn_head_loss(batch_teacher_attn_matrix, batch_student_attn_matrix, teacher, student, v_projection_tags)
    
    for layer in layers:
      loss_ += T.dist(teacher[layer], student[layer],p=2).item()
      
    # find the average loss accross the specified layers.
    avg_loss = loss_/len(layers)

    # the overall loss
    overall_fd_loss = avg_loss + batch_attn_matrix_loss + batch_attn_head_loss
    return overall_fd_loss

  def total_distillation_loss(self, student_target_loss, distillation_loss, fd_loss):
    loss = self.alpha * student_target_loss + (1 - self.alpha) * distillation_loss + fd_loss
    return loss