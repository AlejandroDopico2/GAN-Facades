import torch
import torch.nn.functional as F

def generator_loss(fake_output:torch.Tensor, real_output:torch.Tensor, lambda_l: float = 100):
    gan_loss = F.binary_cross_entropy_with_logits(torch.ones_like(fake_output), fake_output)

    l1_loss = F.l1_loss(real_output, fake_output)

    total_generator_loss = gan_loss + (lambda_l * l1_loss)

    return total_generator_loss, gan_loss, l1_loss
