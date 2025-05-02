import torch
import torch.nn as nn
from .teacher_dit import TeacherDiTEncoder
from .student_dit import StudentDiTEncoder
from .student_decoder import StudentDiTDecoder
from .implicit_discriminator import ImplicitDiscriminator

class IKDDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.teacher = TeacherDiTEncoder(config.teacher)
        self.student_enc = StudentDiTEncoder(config.student)
        self.student_dec = StudentDiTDecoder(config.student)
        self.discriminator = ImplicitDiscriminator(config.student.hidden_dim)

    def forward(self, x, context, mask_ratio, t):
        z_t = self.teacher(x, context)
        z_s = self.student_enc(x, context)
        d_loss = self.discriminator(z_s)
        recon = self.student_dec(z_s)
        return recon, d_loss
