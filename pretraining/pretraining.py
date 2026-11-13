import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


if not hasattr(np, "float"):
    np.float = float


from dataset import build_croma_webdataset, extract_sample_ids, load_allowed_ids
from pretrain_croma import CROMA, get_mask # from https://github.com/antofuller/CROMA
from ssl4eo_s12_dataset import statistics # from https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1


def get_scheduler(optimizer, total_steps, warmup_steps):
    """
    Build a cosine schedule with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        total_steps: Total number of optimization steps.
        warmup_steps: Number of warmup steps.

    Returns:
        A LambdaLR scheduler.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def build_dataloader(dataset, num_workers: int, pin_memory: bool):
    """
    Build the DataLoader for a batched WebDataset.

    Args:
        dataset: Batched WebDataset pipeline.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to use pinned CPU memory.

    Returns:
        A PyTorch DataLoader.
    """
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    return DataLoader(**loader_kwargs)


def next_batch(loader, loader_iter):
    """
    Get the next batch from an iterable DataLoader and restart if needed.

    Args:
        loader: DataLoader instance.
        loader_iter: Current iterator.

    Returns:
        A tuple (batch, loader_iter).
    """
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)

    return batch, loader_iter


def filter_raw_batch_by_ids(
    raw_batch: dict,
    allowed_ids: Optional[set[str]],
) -> Optional[dict]:
    """
    Filter a raw batch by allowed sample IDs.

    Args:
        raw_batch: Raw batch emitted by WebDataset.
        allowed_ids: Optional set of sample IDs to keep.

    Returns:
        Filtered raw batch, or None if all samples are removed.
    """
    if allowed_ids is None:
        return raw_batch

    sample_ids = extract_sample_ids(raw_batch)

    keep_indices = [
        idx for idx, sample_id in enumerate(sample_ids)
        if sample_id in allowed_ids
    ]

    if not keep_indices:
        return None

    if len(keep_indices) == len(sample_ids):
        return raw_batch

    keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long)
    filtered = {}

    for key, value in raw_batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(
            sample_ids
        ):
            filtered[key] = value.index_select(0, keep_indices_tensor)
        elif isinstance(value, list) and len(value) == len(sample_ids):
            filtered[key] = [value[idx] for idx in keep_indices]
        elif isinstance(value, tuple) and len(value) == len(sample_ids):
            filtered[key] = tuple(value[idx] for idx in keep_indices)
        else:
            filtered[key] = value

    return filtered


class GPUCROMAPreprocess:
    """
    Full CROMA preprocessing on GPU.

    Operations:
        - cast to float32
        - flatten temporal dimension
        - concatenate optical and radar channels
        - random square crop with one crop size per batch
        - resize to output size
        - random horizontal flip
        - random vertical flip
        - random 90-degree rotation
        - channel-wise normalization
    """

    def __init__(
        self,
        output_size: int = 120,
        crop_min: int = 60,
        crop_max: int = 180,
    ) -> None:
        self.output_size = output_size
        self.crop_min = crop_min
        self.crop_max = crop_max

        s2_mean = torch.tensor(
            statistics["mean"]["S2L2A"],
            dtype=torch.float32,
        )
        s2_std = torch.tensor(
            statistics["std"]["S2L2A"],
            dtype=torch.float32,
        )
        s1_mean = torch.tensor(
            statistics["mean"]["S1GRD"],
            dtype=torch.float32,
        )
        s1_std = torch.tensor(
            statistics["std"]["S1GRD"],
            dtype=torch.float32,
        )

        self.mean_cpu = torch.cat([s2_mean, s1_mean], dim=0).view(1, -1, 1, 1)
        self.std_cpu = torch.cat([s2_std, s1_std], dim=0).view(1, -1, 1, 1)

    def __call__(self, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """
        Build a final CROMA input batch on GPU.

        Args:
            s2: Tensor of shape [B, T, 12, H, W].
            s1: Tensor of shape [B, T,  2, H, W].

        Returns:
            Tensor of shape [B * T, 14, output_size, output_size].
        """
        if s2.ndim != 5 or s1.ndim != 5:
            raise ValueError(
                "Expected S2 and S1 tensors of shape [B, T, C, H, W], "
                f"got {tuple(s2.shape)} and {tuple(s1.shape)}."
            )

        if s2.shape[0] != s1.shape[0] or s2.shape[1] != s1.shape[1]:
            raise ValueError(
                "S2 and S1 batch/time dimensions do not match. "
                f"Got {tuple(s2.shape)} and {tuple(s1.shape)}."
            )

        s2 = s2.float()
        s1 = s1.float()

        batch_size, n_times, _, height, width = s2.shape

        s2 = s2.reshape(batch_size * n_times, s2.shape[2], height, width)
        s1 = s1.reshape(batch_size * n_times, s1.shape[2], height, width)

        images = torch.cat([s2, s1], dim=1)

        max_valid_crop = min(self.crop_max, height, width)
        min_valid_crop = min(self.crop_min, max_valid_crop)

        if min_valid_crop < 1:
            raise ValueError(
                f"Invalid image shape for cropping: {tuple(images.shape)}"
            )

        crop_size = random.randint(min_valid_crop, max_valid_crop)

        max_top = height - crop_size
        max_left = width - crop_size

        if max_top > 0:
            top = random.randint(0, max_top)
        else:
            top = 0

        if max_left > 0:
            left = random.randint(0, max_left)
        else:
            left = 0

        images = images[:, :, top:top + crop_size, left:left + crop_size]

        images = F.interpolate(
            images,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

        batch_effective = images.shape[0]

        hflip_mask = torch.rand(batch_effective, device=images.device) < 0.5
        if hflip_mask.any():
            images[hflip_mask] = torch.flip(images[hflip_mask], dims=[3])

        vflip_mask = torch.rand(batch_effective, device=images.device) < 0.5
        if vflip_mask.any():
            images[vflip_mask] = torch.flip(images[vflip_mask], dims=[2])

        rotations = torch.randint(
            low=0,
            high=4,
            size=(batch_effective,),
            device=images.device,
        )

        for k in range(4):
            mask = rotations == k
            if mask.any():
                images[mask] = torch.rot90(images[mask], k, dims=[2, 3])

        mean = self.mean_cpu.to(images.device, non_blocking=True)
        std = self.std_cpu.to(images.device, non_blocking=True)
        images = (images - mean) / std

        return images


def main():
    """
    Main DDP training entry point.
    """
    print("Starting main()", flush=True)

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    print(
        f"Initialized DDP: rank={rank}, local_rank={local_rank}, world_size={world_size}",
        flush=True,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = True

    dataset_root = os.path.join(
        os.environ["DSDIR"],
        "HuggingFace",
        "embed2scale",
        "SSL4EO-S12-v1.1",
    )

    raw_batch_size = int(os.environ.get("RAW_BATCH_SIZE", 32))
    steps_per_epoch = int(os.environ.get("STEPS_PER_EPOCH", 1000))
    num_epochs = int(os.environ.get("NUM_EPOCHS", 300))
    num_workers = int(os.environ.get("NUM_WORKERS", 2))
    pin_memory = os.environ.get("PIN_MEMORY", "0") == "1"

    allowed_ids_path: Optional[str] = os.environ.get("ALLOWED_IDS_FILE")
    allowed_ids = load_allowed_ids(allowed_ids_path)

    gpu_preprocess = GPUCROMAPreprocess(output_size=120)

    train_dataset = build_croma_webdataset(
        root=dataset_root,
        raw_batch_size=raw_batch_size,
        split="train",
        shuffle=True,
        shardshuffle=100,
        deterministic=False,
        seed=None,
    )

    loader = build_dataloader(
        train_dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("DataLoader built", flush=True)

    model = CROMA().to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-6,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    if rank == 0:
        work_dir = Path(os.environ["WORK"]) / "checkpoints"
        work_dir.mkdir(parents=True, exist_ok=True)

        log_file = work_dir / "training_log.csv"

        with log_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "step",
                    "loss",
                    "contrastive_loss",
                    "mae_loss",
                    "lr",
                    "effective_batch_size",
                    "step_time",
                    "data_time",
                    "filter_time",
                    "transfer_time",
                    "gpu_pre_time",
                    "forward_time",
                    "backward_time",
                    "opt_time",
                ]
            )

    global_step = 0
    model.train()

    loader_iter = iter(loader)

    for epoch in range(num_epochs):
        steps_done = 0

        while steps_done < steps_per_epoch:
            step_start = time.time()

            data_start = time.time()
            raw_batch, loader_iter = next_batch(loader, loader_iter)
            data_time = time.time() - data_start

            filter_start = time.time()
            raw_batch = filter_raw_batch_by_ids(
                raw_batch=raw_batch,
                allowed_ids=allowed_ids,
            )
            filter_time = time.time() - filter_start

            if raw_batch is None:
                continue

            transfer_start = time.time()
            s2 = raw_batch["S2L2A"].to(device, non_blocking=pin_memory)
            s1 = raw_batch["S1GRD"].to(device, non_blocking=pin_memory)
            transfer_time = time.time() - transfer_start

            gpu_pre_start = time.time()
            imgs = gpu_preprocess(s2=s2, s1=s1)
            gpu_pre_time = time.time() - gpu_pre_start

            bsz = imgs.size(0)

            radar_mask_info = get_mask(
                bsz=bsz,
                seq_len=225,
                device=device,
                mask_ratio=0.75,
            )
            optical_mask_info = get_mask(
                bsz=bsz,
                seq_len=225,
                device=device,
                mask_ratio=0.75,
            )

            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            forward_start = time.time()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                contrastive_loss, mae_loss = model(
                    imgs,
                    radar_mask_info,
                    optical_mask_info,
                    rank=rank,
                    world_size=world_size,
                )
                loss = contrastive_loss + mae_loss

            torch.cuda.synchronize()
            forward_time = time.time() - forward_start

            torch.cuda.synchronize()
            backward_start = time.time()

            loss.backward()

            torch.cuda.synchronize()
            backward_time = time.time() - backward_start

            torch.cuda.synchronize()
            opt_start = time.time()

            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            opt_time = time.time() - opt_start

            step_time = time.time() - step_start
            lr = scheduler.get_last_lr()[0]

            if rank == 0 and global_step % 10 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Step {global_step} | "
                    f"Loss {loss.item():.4f} | "
                    f"Batch {bsz} | "
                    f"step {step_time:.2f}s | "
                    f"data {data_time:.2f}s | "
                    f"filter {filter_time:.2f}s | "
                    f"gpu_pre {gpu_pre_time:.2f}s | "
                    f"fwd {forward_time:.2f}s | "
                    f"bwd {backward_time:.2f}s",
                    flush=True,
                )

                with log_file.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch,
                            global_step,
                            loss.item(),
                            contrastive_loss.item(),
                            mae_loss.item(),
                            lr,
                            bsz,
                            step_time,
                            data_time,
                            filter_time,
                            transfer_time,
                            gpu_pre_time,
                            forward_time,
                            backward_time,
                            opt_time,
                        ]
                    )

            global_step += 1
            steps_done += 1

        if rank == 0:
            checkpoint_path = work_dir / f"croma_epoch_{epoch}.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()