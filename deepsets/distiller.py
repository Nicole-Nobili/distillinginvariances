# Knowledge distillation class from a general teacher model to a student.
# This is configured to use pytorch geometric loader of the ModelNet40 data.

import torch.nn as nn
from torch.nn.functional import softmax
import torch
import numpy as np


class Distiller(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        device: str,
        lr=0.001,
        temp: float = 3.5,
        alpha: float = 0,
    ):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

        self.optimiser = torch.optim.Adam(self.student.parameters(), lr=lr)

        self.student_loss_fn = torch.nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")
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

    def distill(self, train_data, valid_data, epochs):
        for epoch in range(epochs):
            self.student.train()

            student_loss_running = []
            distill_loss_running = []
            total_loss_running = []
            student_accu_running = []
            for data in train_data:
                data = data.to(self.device)
                y_true = data.y.flatten()

                teacher_predictions = self.teacher.predict(x)
                student_predictions = self.student(data.pos)
                student_loss = self.student_loss_fn(student_predictions, y_true)
                distillation_loss = (
                    self.distillation_loss_fn(
                        softmax(teacher_predictions / self.temp, dim=1),
                        softmax(student_predictions / self.temp, dim=1),
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
                    student_predictions.max(dim=1)[1] == y_true
                ) / len(y_true)

                student_loss_running.append(student_loss.item())
                distill_loss_running.append(distillation_loss.item())
                total_loss_running.append(total_loss.item())
                student_accu_running.append(student_accu.item())

            self.all_student_loss_train.append(np.mean(student_loss_running))
            self.all_distill_loss_train.append(np.mean(distill_loss_running))
            self.all_total_loss_train.append(np.mean(total_loss_running))
            self.all_student_accu_train.append(np.mean(student_accu_running))

            student_loss_running = []
            distill_loss_running = []
            total_loss_running = []
            student_accu_running = []
            for data in valid_data:
                data = data.to(device)
                y_true = data.y.flatten()
                y_pred = model.predict(data.pos)

                teacher_predictions = self.teacher.predict(x)
                student_predictions = self.student.predict(data.pos)
                student_loss = self.student_loss_fn(student_predictions, y_true)
                distillation_loss = (
                    self.distillation_loss_fn(
                        softmax(teacher_predictions / self.temp, dim=1),
                        softmax(student_predictions / self.temp, dim=1),
                    )
                    * self.temp**2
                )
                total_loss = (
                    self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                )
                student_accu = torch.sum(
                    student_predictions.max(dim=1)[1] == y_true
                ) / len(y_true)

                student_loss_running.append(student_loss.item())
                distill_loss_running.append(distillation_loss.item())
                total_loss_running.append(total_loss.item())
                student_accu_running.append(student_accu.item())

            self.all_student_loss_valid.append(np.mean(student_loss_running))
            self.all_distill_loss_valid.append(np.mean(distill_loss_running))
            self.all_total_loss_valid.append(np.mean(total_loss_running))
            self.all_student_accu_valid.append(np.mean(student_accu_running))

            print_metrics(epoch, epochs)

    def get_student(self):
        return self.student

    def print_metrics(epoch, epochs):
        """Prints the training and validation metrics in a nice format."""
        print(
            util.tcols.OKGREEN
            + f"Epoch : {epoch + 1}/{epochs}\n"
            + util.tcols.ENDC
            + f"Student train loss = {self.all_student_loss_train[epoch]:.8f}\n"
            + f"Total train loss = {self.all_total_loss_train[epoch]:.8f}\n"
            + f"Student train accu = {self.all_student_accu_train[epoch]:.8f}\n\n"
            + f"Student valid loss = {self.all_student_loss_valid[epoch]:.8f}\n"
            + f"Total valid loss = {self.all_total_loss_valid[epoch]:.8f}\n"
            + f"Student valid accu = {self.all_student_accu_valid[epoch]:.8f}\n"
            + "---"
        )
