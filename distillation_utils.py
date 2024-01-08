import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Distiller(nn.Module):

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device: str,
        lr: float = 0.001,
        epochs: int = 10,
        early_stopping: int = 5,
        temp: float = 3.5,
        alpha: float = 0,
    ):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

        self.optimiser = torch.optim.Adam(self.student.parameters(), lr=lr)

        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.device = device

        self.all_student_loss_train = []
        self.all_distill_loss_train = []
        self.all_total_loss_train = []
        self.all_student_accu_train = []

        self.all_student_loss_valid = []
        self.all_distill_loss_valid = []
        self.all_total_loss_valid = []
        self.all_student_accu_valid = []

        self.alpha = alpha
        self.temp = temp
        self.epochs = epochs
        self.early_stopping = early_stopping

    def distill(self, train_data: DataLoader, valid_data: DataLoader):
        """Distill a teacher into a student model."""
        best_accu = 0
        epochs_no_improve = 0
        self.student.to(self.device)
        self.teacher.to(self.device)

        print("\nDistilling...")
        for epoch in range(self.epochs):
            self.student.train() #may break stuff

            student_loss_running = []
            distill_loss_running = []
            total_loss_running = []
            student_accu_running = []
            for i, (x,y) in enumerate(train_data):
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    self.teacher.eval()
                    try:
                        teacher_predictions = self.teacher(x)
                    except RuntimeError:
                        teacher_predictions = self.teacher(x.view(-1,784))

                student_predictions = self.student(x.view(-1,784))
                student_loss = self.student_loss_fn(student_predictions, y)
                distillation_loss = (
                    self.distillation_loss_fn(
                        nn.functional.log_softmax(teacher_predictions/self.temp, dim=1),
                        nn.functional.log_softmax(student_predictions/self.temp, dim=1),
                    )
                    * self.temp**2
                )
                total_loss = (
                    self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                )

                self.optimiser.zero_grad()
                total_loss.mean().backward()
                self.optimiser.step()
                student_accu = torch.sum(
                    student_predictions.max(dim=1)[1] == y
                ) / len(y)

                student_loss_running.append(student_loss.cpu().item())
                distill_loss_running.append(distillation_loss.cpu().item())
                student_accu_running.append(student_accu.cpu().item())

            self.all_student_loss_train.append(np.mean(student_loss_running))
            self.all_distill_loss_train.append(np.mean(distill_loss_running))
            self.all_student_accu_train.append(np.mean(student_accu_running))

            student_loss_running = []
            distill_loss_running = []
            total_loss_running = []
            student_accu_running = []
            for i, (x,y) in enumerate(valid_data):
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    self.student.eval()
                    self.teacher.eval()
                    try:
                        teacher_predictions = self.teacher(x)
                    except RuntimeError:
                        teacher_predictions = self.teacher(x.view(-1,784))
                    student_predictions = self.student(x.view(-1,784))

                student_loss = self.student_loss_fn(student_predictions, y)
                distillation_loss = (
                    self.distillation_loss_fn(
                        nn.functional.log_softmax(teacher_predictions/self.temp, dim=1),
                        nn.functional.log_softmax(student_predictions/self.temp, dim=1),
                    )
                    * self.temp**2
                )
                total_loss = (
                    self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                )
                student_accu = torch.sum(
                    student_predictions.max(dim=1)[1] == y
                ) / len(y)

                student_loss_running.append(student_loss.cpu().item())
                distill_loss_running.append(distillation_loss.cpu().item())
                student_accu_running.append(student_accu.cpu().item())

            self.all_student_loss_valid.append(np.mean(student_loss_running))
            self.all_distill_loss_valid.append(np.mean(distill_loss_running))
            self.all_student_accu_valid.append(np.mean(student_accu_running))

            if self.all_student_accu_valid[-1] <= best_accu:
                epochs_no_improve += 1
            else:
                best_accu = self.all_student_accu_valid[-1]
                epochs_no_improve = 0

            self.print_metrics(epoch)

        return {
            "student_train_losses": self.all_student_loss_train,
            "student_train_accurs": self.all_student_accu_train,
            "distill_train_losses": self.all_distill_loss_train,
            "student_valid_losses": self.all_student_loss_valid,
            "student_valid_accurs": self.all_student_accu_valid,
            "distill_valid_losses": self.all_distill_loss_valid
        }

    def get_student(self):
        return self.student
    
    def eval_student(self, valid_data):
        accu = []
        loss = []
        for i, (x,y) in enumerate(valid_data):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                self.student.eval()
                student_predictions = self.student(x.view(-1,784))
            student_loss = self.student_loss_fn(student_predictions, y)
            student_accu = torch.sum(
                student_predictions.max(dim=1)[1] == y
            ) / len(y)
            loss.append(student_loss.item())
            accu.append(student_accu.item())

        print(f"Loss : {torch.mean(student_loss)}\n"
            + f"Accuracy: {torch.mean(student_accu)}\n")


    def print_metrics(self, epoch: int):
        """Prints the training and validation metrics in a nice format."""
        print(
            f"Epoch : {epoch + 1}/{self.epochs}\n"
            + f"Student train loss = {self.all_student_loss_train[epoch]:.8f}\n"
            + f"Distill train loss = {self.all_distill_loss_train[epoch]:.8f}\n"
            + f"Student train accu = {self.all_student_accu_train[epoch]:.8f}\n\n"
            + f"Student valid loss = {self.all_student_loss_valid[epoch]:.8f}\n"
            + f"Distill valid loss = {self.all_distill_loss_valid[epoch]:.8f}\n"
            + f"Student valid accu = {self.all_student_accu_valid[epoch]:.8f}\n"
            + "---"
        )
   
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