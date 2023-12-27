import torch.nn as nn
from torch.nn.functional import softmax
import torch
import numpy as np

class Distiller(nn.Module):

    def __init__(self, student: nn.Module, teacher: nn.Module, device: str, load_student_from_path = None, lr = 0.001, temp: float = 3.5, alpha: float = 0):
        #Note that a temperature of 4 is said to work well when the teacher is fairly confident of its predictions
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

        self.optimiser = torch.optim.Adam(self.student.parameters(), lr=lr)

        self.student_loss_fn = torch.nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.device = device

        self.categorical_accuracy = []
        self.student_loss_running = []
        self.distill_loss_running = []
        self.total_loss_running = []

        self.alpha = alpha
        self.temp = temp

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
            with torch.no_grad():
              teacher_predictions = self.teacher(x)

            student_predictions = self.student(x.view(-1, 784))
            student_loss = self.student_loss_fn(student_predictions, y)
            distillation_loss = (
                self.distillation_loss_fn(
                    softmax(teacher_predictions / self.temp, dim=1),
                    softmax(student_predictions / self.temp, dim=1),
                )
                * self.temp**2
            )
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            
            self.optimiser.zero_grad()
            total_loss.mean().backward()
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
