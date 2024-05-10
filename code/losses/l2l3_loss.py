from .mmd import mmd, faster_mmd

def l2l3_loss(divergence_metric, 
              lambda2, 
              lambda3, 
              l2_logits_m1, 
              l2_logits_m2, 
              logits_1, 
              logits_2,
              sigma):
    if divergence_metric == 'mmd':
        L2 = faster_mmd(l2_logits_m1, l2_logits_m2, sigma) * lambda2
        L3 = faster_mmd(logits_1, logits_2, sigma) * lambda3
    elif divergence_metric == 'nothing':
        L2 = 0.
        L3 = 0.

    return L2, L3
