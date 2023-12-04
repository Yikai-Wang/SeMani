import torch
import torch.nn.functional as F
import math


def late_similarity(rep1, rep2, mask_1pad, split=True):
    batch_size1, n_token1, feat_dim = rep1.shape
    batch_size2, n_token2, _ = rep2.shape
    assert batch_size1 == batch_size2
    assert mask_1pad.dim() == 2 and mask_1pad.size(1) == n_token2

    if split:
        output = []
        for b in range(batch_size1):
            rep1_b = rep1[b]
            rep2_b = rep2[b]
            out_i = rep1_b @ rep2_b.t()  # (n1, n2)
            out_i.masked_fill_(mask_1pad[b:b+1], -9.)  # (n1, n2)
            out_i = out_i.max(dim=-1)[0]  # n1
            out_i = out_i.mean(dim=-1)
            output.append(out_i)
        output = torch.stack(output, dim=0)  # bsz
        output = output.mean()
    else:
        output = torch.matmul(rep1.unsqueeze(2), rep2.unsqueeze(1).transpose(-1, -2))  # (b, n1, n2)
        output.masked_fill_(mask_1pad.unsqueeze(1), -9.)
        output = output.max(dim=-1)[0]
        output = output.mean(dim=-1)  # bsz
        output = output.mean()
    return output


def global_clip_loss(clip_model, clip_preprocess, normalized_images, tokenized_text, clip_loss_method,
                     clip_trainer="openai"):
    if clip_loss_method == 'ce':
        raise NotImplementedError

    assert normalized_images.size(0) == tokenized_text.size(0), \
        'Image size {} does not match Text size {}'.format(normalized_images.size(0), tokenized_text.size(0))
    assert normalized_images.min() >= 0.0 and normalized_images.max() <= 1.0, \
        'Images have not been normalized, the minimum is {}, the maximum is {}'.format(
            normalized_images.min(), normalized_images.max()
        )

    normalized_images = clip_preprocess(normalized_images)
    if clip_trainer == "openai":
        clip_img_emb, _ = clip_model.encode_image(normalized_images)
        clip_text_emb, _ = clip_model.encode_text(tokenized_text)
        # normalized features
        clip_img_emb = clip_img_emb / clip_img_emb.norm(dim=-1, keepdim=True)
        clip_text_emb = clip_text_emb / clip_text_emb.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = clip_img_emb @ clip_text_emb.t()  # (img, text)
        assert logits_per_image.size(0) == logits_per_image.size(1)

        total_loss = (1 - torch.diagonal(logits_per_image)).mean()
    else:
        clip_img_emb = clip_model.encode_image(normalized_images)  # (B, S1, DIM)
        clip_text_emb = clip_model.encode_text(tokenized_text)  # (B, S2, DIM)
        # normalized features
        clip_img_emb = clip_img_emb / clip_img_emb.norm(dim=-1, keepdim=True)  # (B, S1, DIM)
        clip_text_emb = clip_text_emb / clip_text_emb.norm(dim=-1, keepdim=True)  # (B, S2, DIM)
        # cosine similarity as logits

        non_padding_txt_token_num = tokenized_text.size(1) - (tokenized_text == 0).sum(dim=1)  # (TXT_BSZ, )
        mask_0pad = (torch.arange(tokenized_text.size(1)).expand_as(tokenized_text).to(tokenized_text)
                     < non_padding_txt_token_num[..., None].expand_as(tokenized_text))
        mask_1pad = ~mask_0pad
        total_loss = late_similarity(clip_img_emb, clip_text_emb, mask_1pad, split=False)
    
    # if clip_loss_method == 'ce':
    #     raise NotImplementedError
    #     labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)
    #
    #     clip_logit_scale = torch.exp(clip_model.logit_scale)
    #     logits_per_text = logits_per_image.t()
    #
    #     logits_per_image = logits_per_image * clip_logit_scale
    #     logits_per_text = logits_per_text * clip_logit_scale
    #
    #     loss_img = F.cross_entropy(logits_per_image, labels)
    #     loss_txt = F.cross_entropy(logits_per_text, labels)
    #     total_loss = (loss_img + loss_txt) / 2
    #
    #     # total_loss = - torch.log(torch.diagonal(logits_per_image)).mean()
    # else:
    #     total_loss = (1 - torch.diagonal(logits_per_image)).mean()
    return total_loss


def fetch_soft_one_hot_map(logits, tau, hard_one_hot, labels=None, num_classes=None, cam_mask=None):
    # logits: Tensor (BSZ, SEQ, SIZE)
    soft_one_hot = F.gumbel_softmax(logits, tau=tau, dim=-1, hard=hard_one_hot)  # (BSZ, SEQ, N_CLS)
    MAP_SIZE = int(math.sqrt(soft_one_hot.size(1)))

    if labels is not None:
        # assert num_classes is not None
        # assert cam_mask is not None
        # labels_one_hot = F.one_hot(labels, num_classes=num_classes).to(soft_one_hot)  # (BSZ, SEQ, N_CLS)
        # soft_one_hot = save_assign(tgt_tensor=soft_one_hot, src_tensor=labels_one_hot, obj_indices=~cam_mask)
        raise NotImplementedError

    soft_one_hot = soft_one_hot.view(
        soft_one_hot.size(0),
        MAP_SIZE,
        MAP_SIZE,
        -1
    )  # (BSZ, H, W, N_CLS)
    soft_one_hot = soft_one_hot.permute(0, 3, 1, 2)  # (BSZ, N_CLS, H, W)

    return soft_one_hot

