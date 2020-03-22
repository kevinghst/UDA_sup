import pdb

import torch
import numpy as np
import torch.nn.functional as F


def get_mixmatch_loss_two(cfg, model, sup_batch, unsup_batch, global_step):
    input_ids, segment_ids, input_mask, label_ids = sup_batch
    if unsup_batch:
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

    batch_size = input_ids.shape[0]
    sup_size = label_ids.shape[0]

    with torch.no_grad():
        # compute guessed labels of unlabel samples
        outputs_u = model(input_ids=ori_input_ids, segment_ids=ori_segment_ids, input_mask=ori_input_mask)
        outputs_u2 = model(input_ids=aug_input_ids, segment_ids=aug_segment_ids, input_mask=aug_input_mask)
        p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        pt = p**(1/cfg.uda_softmax_temp)
        targets_u = pt / pt.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()
        targets_u = torch.cat((targets_u, targets_u), dim=0)

        # confidence-based masking
        if cfg.uda_confidence_thresh != -1:
            unsup_loss_mask = torch.max(targets_u, dim=-1)[0] > cfg.uda_confidence_thresh
            unsup_loss_mask = unsup_loss_mask.type(torch.float32)
        else:
            unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
        unsup_loss_mask = unsup_loss_mask.to(_get_device())

    input_ids = torch.cat((input_ids, ori_input_ids, aug_input_ids), dim=0)
    seg_ids = torch.cat((segment_ids, ori_segment_ids, aug_segment_ids), dim=0)
    input_mask = torch.cat((input_mask, ori_input_mask, aug_input_mask), dim=0)

    logits = model(input_ids, seg_ids, input_mask)

    logits_x = logits[:sup_size]
    logits_u = logits[sup_size:]

    sup_loss = sup_criterion(logits_x, label_ids)
    if cfg.tsa:
        tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
        # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
        loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
    else:
        sup_loss = torch.mean(sup_loss)

    log_probs_u = F.log_softmax(logits_u, dim=1)
    unsup_loss = torch.sum(unsup_criterion(log_probs_u, targets_u), dim=-1)
    unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())

    final_loss = sup_loss + cfg.uda_coeff*unsup_loss
    return final_loss, sup_loss, unsup_loss
        


def get_mixmatch_loss(cfg, model, sup_batch, unsup_batch, global_step):
    input_ids, segment_ids, input_mask, label_ids = sup_batch
    if unsup_batch:
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

    batch_size = input_ids.shape[0]

    # Transform label to one-hot
    label_ids = torch.zeros(batch_size, 2).scatter_(1, label_ids.cpu().view(-1,1), 1).cuda()

    with torch.no_grad():
        # compute guessed labels of unlabel samples
        outputs_u = model(input_ids=ori_input_ids, segment_ids=ori_segment_ids, input_mask=ori_input_mask)
        outputs_u2 = model(input_ids=aug_input_ids, segment_ids=aug_segment_ids, input_mask=aug_input_mask)
        p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        pt = p**(1/cfg.T)
        targets_u = pt / pt.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()

    concat_input_ids = [input_ids, ori_input_ids, aug_input_ids]
    concat_seg_ids = [segment_ids, ori_segment_ids, aug_segment_ids]
    concat_input_mask = [input_mask, ori_input_mask, aug_input_mask]
    concat_targets = [label_ids, targets_u, targets_u]

    # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
    int_input_ids = interleave(concat_input_ids, batch_size)
    int_seg_ids = interleave(concat_seg_ids, batch_size)
    int_input_mask = interleave(concat_input_mask, batch_size)
    int_targets = interleave(concat_targets, batch_size)

    h_zero = model(
        input_ids=int_input_ids[0],
        segment_ids=int_seg_ids[0],
        input_mask=int_input_mask[0], 
        output_h=True
    )

    h_one = model(
        input_ids=int_input_ids[1],
        segment_ids=int_seg_ids[1],
        input_mask=int_input_mask[1], 
        output_h=True
    )

    h_two = model(
        input_ids=int_input_ids[2],
        segment_ids=int_seg_ids[2],
        input_mask=int_input_mask[2], 
        output_h=True
    )

    int_h = torch.cat([h_zero, h_one, h_two], dim=0)
    int_targets = torch.cat([int_targets[0], int_targets[1], int_targets[2]])

    l = np.random.beta(cfg.alpha, cfg.alpha)
    l = max(l, 1-l)

    idx = torch.randperm(int_h.size(0))

    h_a, h_b = int_h, int_h[idx]
    target_a, target_b = int_targets, int_targets[idx]

    mixed_int_h = l * h_a + (1 - l) * h_b
    mixed_int_target = l * target_a + (1 - l) * target_b

    mixed_int_h = list(torch.split(mixed_int_h, batch_size))
    mixed_int_targets = list(torch.split(mixed_int_target, batch_size))

    logits_one = model(input_h=mixed_int_h[0])
    logits_two = model(input_h=mixed_int_h[1])
    logits_three = model(input_h=mixed_int_h[2])

    logits = [logits_one, logits_two, logits_three]


    # put interleaved samples back
    logits = interleave(logits, batch_size)
    targets = interleave(mixed_int_targets, batch_size)

    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)

    targets_x = targets[0]
    targets_u = torch.cat(targets[1:], dim=0)

    #Lx, Lu, w = train_criterion(logits_x, targets_x, logits_u, targets_u, epoch+batch_idx/cfg.val_iteration)
    Lx, Lu, w = train_criterion(logits_x, targets_x, logits_u, targets_u, global_step, cfg.lambda_u, cfg.total_steps)

    final_loss = Lx + w * Lu

    return final_loss, Lx, Lu

def get_uda_mixup_loss(cfg, model, sup_batch, unsup_batch, global_step):
    # logits -> prob(softmax) -> log_prob(log_softmax)

    # batch
    input_ids, segment_ids, input_mask, og_label_ids = sup_batch

    sup_size = input_ids.size(0)

    # convert label_ids to hot vector
    label_ids = torch.zeros(sup_size, 2).scatter_(1, og_label_ids.cpu().view(-1,1), 1).cuda()

    if unsup_batch:
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

        input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
        segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
        input_mask = torch.cat((input_mask, aug_input_mask), dim=0)

    # logits
    hidden = model(
        input_ids=input_ids,
        segment_ids=segment_ids,
        input_mask=input_mask,
        output_h=True
    )

    sup_hidden = hidden[:sup_size]
    unsup_hidden = hidden[sup_size:]

    l = np.random.beta(cfg.alpha, cfg.alpha)
    l = max(l, 1-l)
    idx = torch.randperm(sup_size)
    sup_h_a, sup_h_b = sup_hidden, sup_hidden[idx]
    sup_label_a, sup_label_b = label_ids, label_ids[idx]

    mixed_sup_h = l * sup_h_a + (1 - l) * sup_h_b
    mixed_sup_label = l * sup_label_a + (1 - l) * sup_label_b

    hidden = torch.cat([mixed_sup_h, unsup_hidden], dim=0)

    logits = model(input_h=hidden)

    sup_logits = logits[:sup_size]
    unsup_logits = logits[sup_size:]

    # sup loss
    sup_loss = -torch.sum(F.log_softmax(sup_logits, dim=1) * mixed_sup_label, dim=1)

    if cfg.tsa:
        tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
        # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
        loss_mask = torch.ones_like(og_label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
    else:
        sup_loss = torch.mean(sup_loss)

    # unsup loss
    if unsup_batch:
        # ori
        with torch.no_grad():
            ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
            ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target
            # ori_log_prob = F.log_softmax(ori_logits, dim=-1)

            # confidence-based masking
            if cfg.uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
            unsup_loss_mask = unsup_loss_mask.to(_get_device())

        # aug
        # softmax temperature controlling
        uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
        aug_log_prob = F.log_softmax(unsup_logits / uda_softmax_temp, dim=-1)

        # KLdiv loss
        """
            nn.KLDivLoss (kl_div)
            input : log_prob (log_softmax)
            target : prob    (softmax)
            https://pytorch.org/docs/stable/nn.html

            unsup_loss is divied by number of unsup_loss_mask
            it is different from the google UDA official
            The official unsup_loss is divided by total
            https://github.com/google-research/uda/blob/master/text/uda.py#L175
        """
        unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())
        final_loss = sup_loss + cfg.uda_coeff*unsup_loss

        return final_loss, sup_loss, unsup_loss
    return sup_loss, None, None

def get_loss(cfg, model, sup_batch, unsup_batch, global_step):
    # logits -> prob(softmax) -> log_prob(log_softmax)

    # batch
    input_ids, segment_ids, input_mask, label_ids = sup_batch
    if unsup_batch:
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask = unsup_batch

        input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
        segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
        input_mask = torch.cat((input_mask, aug_input_mask), dim=0)
            
    # logits
    hidden = model(
        input_ids=input_ids, 
        segment_ids=segment_ids, 
        input_mask=input_mask,
        output_h=True
    )
    logits = model(input_h=hidden)

    # sup loss
    sup_size = label_ids.shape[0]            
    sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
    if cfg.tsa:
        tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
        # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
        loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
    else:
        sup_loss = torch.mean(sup_loss)

    # unsup loss
    if unsup_batch:
        # ori
        with torch.no_grad():
            ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
            ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target
            # temp control
            #ori_prob = ori_prob**(1/cfg.uda_softmax_temp)

            # confidence-based masking
            if cfg.uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
            unsup_loss_mask = unsup_loss_mask.to(_get_device())
                    
        # aug
        uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
        aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

        # KLdiv loss
        """
            nn.KLDivLoss (kl_div)
            input : log_prob (log_softmax)
            target : prob    (softmax)
            https://pytorch.org/docs/stable/nn.html

            unsup_loss is divied by number of unsup_loss_mask
            it is different from the google UDA official
            The official unsup_loss is divided by total
            https://github.com/google-research/uda/blob/master/text/uda.py#L175
        """
        unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())

        final_loss = sup_loss + cfg.uda_coeff*unsup_loss

        return final_loss, sup_loss, unsup_loss
    return sup_loss, None, None