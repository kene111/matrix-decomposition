import torch as T
from torch import nn

class KnowledgeDistillation(nn.Module):
  def __init__(self, teacher, student, loss, feature_layers):
    super().__init__()
    self.teacher_model = teacher.eval()
    self.student_model = student.train()
    self.loss = loss
    self.student_optimizer = T.optim.AdamW(self.student_model.parameters(),lr=0.0001)
    self.feature_layers = feature_layers

  def teacher_inference(self, batch):
    output = self.teacher_model(**batch)
    return output

  def _train_student(self, batch):

    # teachers outputs
    teachers_outputs = self.teacher_inference(batch)
 
    # student predictions
    s_outputs = self.student_model(**batch)


    student_logits = s_outputs["logits"].reshape(-1, s_outputs["logits"].shape[-1])
    labels = batch['labels'].reshape(-1)

    student_target_loss = self.loss.get_crossentropy_loss(student_logits, labels)
    # response based distillation 
    kd_loss = self.loss.get_kd_loss(s_outputs.logits, teachers_outputs.logits)
    # feature based distillation
    fd_loss = self.loss.get_fd_loss(self.teacher_model.state_dict(), self.student_model.state_dict(), self.feature_layers)
    
    overall_loss = self.loss.total_distillation_loss(student_target_loss, kd_loss, fd_loss)

    self.student_optimizer.zero_grad()
    overall_loss.backward()
    
    self.student_optimizer.step()

    return overall_loss

  def forward(self, batch_data):
    return self._train_student(batch_data)