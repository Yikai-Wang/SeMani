import torch
import os
import sys
import random
import torchvision.transforms as T
sys.path.append("../")
import clip_openai


from dalle_pytorch.clip_loss import fetch_soft_one_hot_map, global_clip_loss



def reduce_all_tensor(tensor, world_size):
    averaged = tensor.detach().clone()
    torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
    return averaged / world_size


@torch.no_grad()
def validation(model, loader, args, world_size, vae_model, clip_model, clip_img_preprocess):
    model.eval()

    total_loss = 0.0
    total_loss_text = 0.0
    total_loss_image = 0.0
    total_loss_meaningful_text = 0.0
    total_loss_clip = 0.0

    for i, (text, images, original_text, image_name) in enumerate(loader):
        # inputs
        text, images = map(lambda t: t.cuda(), (text, images))
        # img tokens
        if images.dim() == 4:
            if args.fp16:
                images = images.half()
            images = vae_model.get_codebook_indices(images)
        else:
            images = images

        loss, (loss_text, loss_image), loss_meaningful_text, img_logits = model(text, images, return_loss=True)

        # CLIP LOSS
        clip_tokenized_texts = clip_openai.tokenize(original_text, truncate=True).to(images.device)
        soft_one_hot = fetch_soft_one_hot_map(
            logits=img_logits, tau=args.gumbel_tau, hard_one_hot=args.straight_through,
            labels=None
        )
        new_img = vae_model(soft_one_hot)
        loss_clip = global_clip_loss(
            clip_model, clip_img_preprocess, new_img, clip_tokenized_texts,
            clip_loss_method=args.clip_loss_method, clip_trainer=args.clip_trainer
        )

        loss = loss + loss_clip * args.clip_loss_weight

        if world_size > 1:
            total_loss += reduce_all_tensor(loss, world_size).item()
            total_loss_text += reduce_all_tensor(loss_text, world_size).item()
            total_loss_image += reduce_all_tensor(loss_image, world_size).item()
            total_loss_meaningful_text += reduce_all_tensor(loss_meaningful_text, world_size).item()
            total_loss_clip += reduce_all_tensor(loss_clip, world_size).item()
        else:
            total_loss += loss.item()
            total_loss_text += loss_text.item()
            total_loss_image += loss_image.item()
            total_loss_meaningful_text += loss_meaningful_text.item()
            total_loss_clip += loss_clip.item()

    total_loss /= len(loader)
    total_loss_text /= len(loader)
    total_loss_image /= len(loader)
    total_loss_meaningful_text /= len(loader)
    total_loss_clip /= len(loader)

    model.train()
    return total_loss, (total_loss_text, total_loss_image), total_loss_meaningful_text, total_loss_clip


@torch.no_grad()
def validation_finetune(model, loader, args, world_size, vae_model, clip_model, clip_img_preprocess):
    model.eval()

    total_loss = 0.0
    total_loss_text = 0.0
    total_loss_image = 0.0
    total_loss_vc = 0.0
    total_loss_meaningful_text = 0.0
    total_loss_clip = 0.0

    for i, (text, images, original_text, image_name) in enumerate(loader):
        # inputs
        text, images = map(lambda t: t.cuda(), (text, images))
        # img tokens
        bsz = text.size(0)
        use_context = torch.zeros(bsz, device=text.device, dtype=torch.bool)
        use_context_id = random.sample(range(bsz), max(bsz//2, 1))
        use_context[use_context_id] = True
        if images.dim() == 4:
            visual_context = T.functional.rgb_to_grayscale(images, num_output_channels=3)
            if args.fp16:
                visual_context = vae_model.get_codebook_indices(visual_context.half())
            else:
                visual_context = vae_model.get_codebook_indices(visual_context)

            if args.fp16:
                images = vae_model.get_codebook_indices(images.half())
            else:
                images = vae_model.get_codebook_indices(images)
        else:
            images = images

        loss, (loss_text, loss_vc, loss_image), loss_meaningful_text, img_logits = \
            model(text, images, visual_context, use_context, return_loss=True)

        # CLIP LOSS
        clip_tokenized_texts = clip_openai.tokenize(original_text, truncate=True).to(images.device)
        soft_one_hot = fetch_soft_one_hot_map(
            logits=img_logits, tau=args.gumbel_tau, hard_one_hot=args.straight_through,
            labels=None
        )
        new_img = vae_model(soft_one_hot)
        loss_clip = global_clip_loss(
            clip_model, clip_img_preprocess, new_img, clip_tokenized_texts,
            clip_loss_method=args.clip_loss_method, clip_trainer=args.clip_trainer
        )

        loss = loss + loss_clip * args.clip_loss_weight

        if world_size > 1:
            total_loss += reduce_all_tensor(loss, world_size).item()
            total_loss_text += reduce_all_tensor(loss_text, world_size).item()
            total_loss_image += reduce_all_tensor(loss_image, world_size).item()
            total_loss_vc += reduce_all_tensor(loss_vc, world_size).item()
            total_loss_meaningful_text += reduce_all_tensor(loss_meaningful_text, world_size).item()
            total_loss_clip += reduce_all_tensor(loss_clip, world_size).item()
        else:
            total_loss += loss.item()
            total_loss_text += loss_text.item()
            total_loss_image += loss_image.item()
            total_loss_vc += loss_vc.item()
            total_loss_meaningful_text += loss_meaningful_text.item()
            total_loss_clip += loss_clip.item()

    total_loss /= len(loader)
    total_loss_text /= len(loader)
    total_loss_image /= len(loader)
    total_loss_vc /= len(loader)
    total_loss_meaningful_text /= len(loader)
    total_loss_clip /= len(loader)

    model.train()
    return total_loss, (total_loss_text, total_loss_vc, total_loss_image), total_loss_meaningful_text, total_loss_clip


@torch.no_grad()
def validation_vae(model, loader, args, world_size, vae_model, clip_model, clip_img_preprocess, perceptual_model):
    vae_model.eval()

    total_loss = 0.0
    total_loss_pixel = 0.0
    total_loss_perceptual = 0.0
    total_loss_clip = 0.0

    for i, (text, images, original_text, image_name) in enumerate(loader):
        # inputs
        text, images = map(lambda t: t.cuda(), (text, images))
        # img tokens
        if images.dim() == 4:
            original_images = images.clone()
            if args.fp16:
                if hasattr(vae_model, "module"):
                    images = vae_model.module.get_codebook_indices(images.half())
                else:
                    images = vae_model.get_codebook_indices(images.half())
            else:
                if hasattr(vae_model, "module"):
                    images = vae_model.module.get_codebook_indices(images)
                else:
                    images = vae_model.get_codebook_indices(images)
        else:
            raise RuntimeError

        img_logits = model(text, images, return_loss=False)

        # CLIP LOSS
        clip_tokenized_texts = clip_openai.tokenize(original_text, truncate=True).to(images.device)
        soft_one_hot = fetch_soft_one_hot_map(
            logits=img_logits, tau=args.gumbel_tau, hard_one_hot=args.straight_through,
            labels=None
        )
        new_img = vae_model(soft_one_hot)
        loss_clip = global_clip_loss(
            clip_model, clip_img_preprocess, new_img, clip_tokenized_texts,
            clip_loss_method=args.clip_loss_method, clip_trainer=args.clip_trainer
        )

        loss_perceptual = perceptual_model(original_images, new_img).mean()
        loss_pixel = torch.abs(new_img - original_images).mean()

        loss = loss_clip + args.loss_pixel_weight * loss_pixel + args.loss_perceptual_weight * loss_perceptual

        if world_size > 1:
            total_loss += reduce_all_tensor(loss, world_size).item()
            total_loss_pixel += reduce_all_tensor(loss_pixel, world_size).item()
            total_loss_perceptual += reduce_all_tensor(loss_perceptual, world_size).item()
            total_loss_clip += reduce_all_tensor(loss_clip, world_size).item()
        else:
            total_loss += loss.item()
            total_loss_pixel += loss_pixel.item()
            total_loss_perceptual += loss_perceptual.item()
            total_loss_clip += loss_clip.item()

    total_loss /= len(loader)
    total_loss_pixel /= len(loader)
    total_loss_perceptual /= len(loader)
    total_loss_clip /= len(loader)

    vae_model.train()
    return total_loss, total_loss_pixel, total_loss_perceptual, total_loss_clip

