import torch.nn as nn
from torch.nn.functional import softmax
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

class Distiller(nn.Module):

    def __init__(self, student: nn.Module, teacher: nn.Module, device: str, load_student_from_path = None, lr = 0.001, temp: float = 3.5, alpha: float = 0):
        #Note that a temperature of 4 is said to work well when the teacher is fairly confident of its predictions
        super(Distiller, self).__init__()
        print("hfafu")
        self.student = student
        self.teacher = teacher

        self.optimiser = torch.optim.Adam(self.student.parameters(), lr=lr)

        self.student_loss_fn = torch.nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.device = device

        self.categorical_accuracy = []
        self.student_loss_running = []
        self.distill_loss_running = []
        self.total_loss_running = []

        self.alpha = alpha
        self.temp = 1

        if load_student_from_path is not None:
          state_dict = torch.load(load_student_from_path)
          self.student.load_state_dict(state_dict)

    def distill(self, train_dataloader, epochs, save_path_folder: None):
        self.student_loss_running = []
        self.distill_loss_running = []
        self.total_loss_running = []
        for epoch in range(epochs):
          """Train the student network through one feed forward."""

          for i, (x, y)  in enumerate(train_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad(): #only needed for compute time reasons
              try:
                teacher_predictions = self.teacher(x)
              except RuntimeError:
                  teacher_predictions = self.teacher(x.view(-1, 784))

            student_predictions = self.student(x.view(-1, 784))
            student_loss = self.student_loss_fn(student_predictions, y)
            isnan = torch.isnan(student_loss).any().item()
            if isnan:
               print("student loss isnan")
            #print(teacher_predictions[0])
            input = nn.functional.softmax(teacher_predictions/self.temp, dim=1)
            #print(input[0])
            input = nn.functional.log_softmax(teacher_predictions/self.temp, dim=1)
            #print(input[0])
            hasnan = torch.isnan(input).any().item()
            if hasnan == True:
               print("input hasnan")
            hasnan = torch.isnan(student_predictions).any().item()
            if hasnan == True:
               print("student output hasnan")
            #print(student_predictions[0])
            target = nn.functional.softmax(student_predictions/self.temp, dim=1)
            #print(target[0])
            target = nn.functional.log_softmax(student_predictions/self.temp, dim=1)
            #print(target[0])
            hasnan = torch.isnan(target).any().item()
            if hasnan == True:
               print("target hasnan")
            distillation_loss = self.distillation_loss_fn(
                        input = input,
                        target = target,
                    )* self.temp**2
            total_loss = distillation_loss
            #total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            #print(total_loss)
            hasnan = torch.isnan(total_loss).any().item()
            if hasnan == True:
               print("loss hasnan")

            self.optimiser.zero_grad()
            total_loss.mean().backward()
            #clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimiser.step()

            if (i + 1) % 100 == 0:
              print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Student Loss : {student_loss.item():.4f}, Total Loss: {total_loss.item():.4f}')

              self.student_loss_running.append(student_loss.item())
              self.distill_loss_running.append(distillation_loss.item())
              self.total_loss_running.append(total_loss.item())

        student_loss = np.mean(self.student_loss_running)
        distill_loss = np.mean(self.distill_loss_running)
        total_loss = np.mean(self.total_loss_running)

        print(f"Student loss: {student_loss}")
        print(f"Distillation loss: {distill_loss}")
        print(f"Total loss: {total_loss}")

        if save_path_folder is not None:
          save_path = save_path_folder + 'distiller'
          torch.save(self.student.state_dict(), save_path)
          print('saved model')
      
    def get_student(self):
       return self.student

    def test_step(self, test_loader):
        """Test the student network."""
        with torch.no_grad():
          correct = 0
          total = 0

          for x,y in test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_prediction = self.student(x.view(-1,784))
            _, predicted = torch.max(y_prediction.data,1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
          accuracy = correct/total
          print(f'Test Accuracy: {accuracy:.4f}')
   
    def compute_fidelity(self, test_loader):
       """Compute the student-teacher average top-1 agreement and the KL divergence."""
       kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)
       top1_agreement_running = []
       kldiv_loss_running = []
       teacher = self.teacher
       student = self.student
       for x,_ in test_loader:
          y_teacher = teacher(x)
          y_student = student(x)
          top1_agreement = torch.sum(
                y_student.max(dim=1)[1] == y_teacher.max(dim=1)[1]
            ) / len(y_student)
          kldiv_loss = kldiv(
                nn.functional.log_softmax(y_teacher, dim=1),
                nn.functional.log_softmax(y_student, dim=1)
            )
          top1_agreement_running.append(top1_agreement.cpu().item())
          kldiv_loss_running.append(kldiv_loss.cpu().item())

       valid_top1_agreement = np.mean(top1_agreement_running)
       valid_kldiv_loss = np.mean(kldiv_loss_running)

       return {
         "top1_agreement": valid_top1_agreement,
         "teach_stu_kldiv": valid_kldiv_loss
       }