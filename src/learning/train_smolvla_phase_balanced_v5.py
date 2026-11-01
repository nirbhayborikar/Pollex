"""
Phase-Balanced SmolVLA V5 Fine-Tuning with H=20.

V1:
    Learned grasp and sustained lift, but severely over-lifted.

V2:
    Added stronger lower/release supervision, but suppressed lift.

V3:
    Introduced temporal semantic phases instead of classifying post-grasp
    frames from instantaneous dz sign.

V4:
    Reduced action horizon to H=10. Chunk purity improved substantially,
    but the horizon was too short and autonomous approach failed.

V5:
    Uses the same temporal phase-balanced supervision as V3, but with H=20
    as a compromise between:
        H=50 -> excessive phase contamination
        H=10 -> insufficient long-horizon behavior
        H=20 -> intermediate phase purity and temporal context

Target sampling:
    hover                 5%
    approach             10%
    descent/pre-grasp    15%
    grasp closing        10%
    post-grasp lift      25%
    post-grasp lower     12%
    release              13%
    retract              10%

Training:
    action horizon H = 20
    max action dim   = 32

Output:
    outputs/smolvla_glass_phase_balanced_v5/
"""

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector


DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)

CHUNK_SIZE = 20
MAX_ACTION_DIM = 32
RELEASE_SETTLE_FRAMES = 12

PHASE_TARGETS = {
    "hover": 0.05,
    "approach": 0.10,
    "descent_pregrasp": 0.15,
    "grasp_closing": 0.10,
    "postgrasp_lift": 0.25,
    "postgrasp_lower": 0.12,
    "release": 0.13,
    "retract": 0.10,
}


def classify_phase(
    frame: int,
    hover_end: int,
    descent_start: int,
    grasp_trigger_frame: int,
    physical_grasp_frame: int,
    peak_lift_frame: int,
    release_frame: int,
    retract_frame: int,
) -> str:
    """Classify one expert frame using temporal task boundaries."""

    if frame < hover_end:
        return "hover"
    if frame < descent_start:
        return "approach"
    if frame < grasp_trigger_frame:
        return "descent_pregrasp"
    if frame < physical_grasp_frame:
        return "grasp_closing"
    if frame < peak_lift_frame:
        return "postgrasp_lift"
    if frame < release_frame:
        return "postgrasp_lower"
    if frame < retract_frame:
        return "release"

    return "retract"


class PhaseBalancedGlassDataset(Dataset):
    """RGB + 8D state + H-step action chunk + temporal phase."""

    def __init__(
        self,
        npz_paths: List[str],
        expert_replay_path: str,
        chunk_size: int = CHUNK_SIZE,
        max_action_dim: int = MAX_ACTION_DIM,
    ):
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.samples = []

        replay_path = Path(expert_replay_path)

        if not replay_path.exists():
            raise FileNotFoundError(
                f"Expert replay not found: {replay_path}"
            )

        with np.load(replay_path, allow_pickle=False) as replay:
            replay_actions = np.asarray(
                replay["executed_actions"],
                dtype=np.float32,
            )

            replay_tumbler = np.asarray(
                replay["tumbler_positions"],
                dtype=np.float32,
            )

            grasp_trigger_frame = int(
                np.asarray(
                    replay["grasp_trigger_frame"]
                ).item()
            )

            physical_grasp_frame = int(
                np.asarray(
                    replay["physical_grasp_latch_frame"]
                ).item()
            )

        if replay_actions.ndim != 2 or replay_actions.shape[1] != 7:
            raise ValueError(
                f"Expected replay actions (T,7), "
                f"got {replay_actions.shape}"
            )

        if replay_tumbler.shape != (
            len(replay_actions),
            3,
        ):
            raise ValueError(
                f"Expected tumbler positions (T,3), "
                f"got {replay_tumbler.shape}"
            )

        if not np.isfinite(replay_actions).all():
            raise ValueError(
                "Expert actions contain NaN or infinity."
            )

        if not np.isfinite(replay_tumbler).all():
            raise ValueError(
                "Expert tumbler states contain NaN or infinity."
            )

        frame_indices = np.arange(
            len(replay_actions)
        )

        postgrasp_open = np.where(
            (frame_indices > physical_grasp_frame)
            & (replay_actions[:, 6] <= 0.0)
        )[0]

        if len(postgrasp_open) == 0:
            raise RuntimeError(
                "No release transition found "
                "after physical grasp."
            )

        release_frame = int(
            postgrasp_open[0]
        )

        retract_frame = min(
            len(replay_actions),
            release_frame + RELEASE_SETTLE_FRAMES,
        )

        if release_frame <= physical_grasp_frame:
            raise RuntimeError(
                f"Invalid release boundary: "
                f"physical grasp={physical_grasp_frame}, "
                f"release={release_frame}"
            )

        lift_window = replay_tumbler[
            physical_grasp_frame:release_frame,
            2,
        ]

        if len(lift_window) == 0:
            raise RuntimeError(
                "No samples found between grasp and release."
            )

        peak_lift_frame = (
            physical_grasp_frame
            + int(np.argmax(lift_window))
        )

        if not (
            physical_grasp_frame
            < peak_lift_frame
            < release_frame
        ):
            raise RuntimeError(
                f"Invalid boundaries: "
                f"physical_grasp={physical_grasp_frame}, "
                f"peak_lift={peak_lift_frame}, "
                f"release={release_frame}"
            )

        self.boundaries = {
            "grasp_trigger_frame": grasp_trigger_frame,
            "physical_grasp_frame": physical_grasp_frame,
            "peak_lift_frame": peak_lift_frame,
            "release_frame": release_frame,
            "retract_frame": retract_frame,
        }

        initial_tumbler_z = float(
            replay_tumbler[0, 2]
        )

        peak_tumbler_z = float(
            replay_tumbler[
                peak_lift_frame,
                2,
            ]
        )

        print(
            f"[*] Expert boundaries: "
            f"grasp_trigger={grasp_trigger_frame}, "
            f"physical_grasp={physical_grasp_frame}, "
            f"peak_lift={peak_lift_frame}, "
            f"release={release_frame}, "
            f"retract={retract_frame}"
        )

        print(
            f"[*] Expert peak lift: "
            f"{(peak_tumbler_z - initial_tumbler_z) * 100.0:.2f} cm"
        )

        global_frame_offset = 0

        for npz_path in npz_paths:
            path = Path(npz_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Training episode not found: {path}"
                )

            with np.load(
                path,
                allow_pickle=False,
            ) as data:
                images = np.asarray(
                    data["images"],
                    dtype=np.uint8,
                )

                actions = np.asarray(
                    data["actions"],
                    dtype=np.float32,
                )

                states = np.asarray(
                    data["state_8d"],
                    dtype=np.float32,
                )

            number_of_frames = len(actions)

            if images.shape != (
                number_of_frames,
                256,
                256,
                3,
            ):
                raise ValueError(
                    f"Invalid image shape: "
                    f"{images.shape}"
                )

            if states.shape != (
                number_of_frames,
                8,
            ):
                raise ValueError(
                    f"Invalid state shape: "
                    f"{states.shape}"
                )

            if actions.shape != (
                number_of_frames,
                7,
            ):
                raise ValueError(
                    f"Invalid action shape: "
                    f"{actions.shape}"
                )

            if not np.isfinite(states).all():
                raise ValueError(
                    f"Non-finite states in {path}"
                )

            if not np.isfinite(actions).all():
                raise ValueError(
                    f"Non-finite actions in {path}"
                )

            if len(npz_paths) == 1:
                if replay_actions.shape != actions.shape:
                    raise ValueError(
                        f"Training/replay action shape mismatch: "
                        f"{actions.shape} != "
                        f"{replay_actions.shape}"
                    )

                if not np.allclose(
                    actions,
                    replay_actions,
                    atol=1e-5,
                ):
                    max_difference = float(
                        np.max(
                            np.abs(
                                actions
                                - replay_actions
                            )
                        )
                    )

                    raise ValueError(
                        "Training episode does not contain "
                        "Iteration 11 actions. "
                        f"Maximum difference: "
                        f"{max_difference:.6f}"
                    )

            hover_end = min(
                40,
                number_of_frames,
            )

            descent_start = min(
                86,
                number_of_frames,
            )

            grasp_trigger = int(
                np.clip(
                    grasp_trigger_frame,
                    0,
                    number_of_frames,
                )
            )

            physical_grasp = int(
                np.clip(
                    physical_grasp_frame,
                    0,
                    number_of_frames,
                )
            )

            episode_peak = int(
                np.clip(
                    peak_lift_frame,
                    0,
                    number_of_frames,
                )
            )

            episode_release = int(
                np.clip(
                    release_frame,
                    0,
                    number_of_frames,
                )
            )

            episode_retract = int(
                np.clip(
                    retract_frame,
                    0,
                    number_of_frames,
                )
            )

            for frame in range(
                number_of_frames
            ):
                future_length = min(
                    chunk_size,
                    number_of_frames - frame,
                )

                action_chunk = np.zeros(
                    (
                        chunk_size,
                        max_action_dim,
                    ),
                    dtype=np.float32,
                )

                action_chunk[
                    :future_length,
                    :7,
                ] = actions[
                    frame:
                    frame + future_length
                ]

                if future_length < chunk_size:
                    action_chunk[
                        future_length:,
                        :7,
                    ] = actions[
                        frame
                        + future_length
                        - 1
                    ]

                phase = classify_phase(
                    frame=frame,
                    hover_end=hover_end,
                    descent_start=descent_start,
                    grasp_trigger_frame=grasp_trigger,
                    physical_grasp_frame=physical_grasp,
                    peak_lift_frame=episode_peak,
                    release_frame=episode_release,
                    retract_frame=episode_retract,
                )

                self.samples.append({
                    "image": images[
                        frame
                    ].copy(),
                    "state": states[
                        frame
                    ].copy(),
                    "action_chunk": (
                        action_chunk
                    ),
                    "frame": (
                        global_frame_offset
                        + frame
                    ),
                    "phase": phase,
                })

            global_frame_offset += (
                number_of_frames
            )

        if not self.samples:
            raise RuntimeError(
                "No training samples loaded."
            )

        self.phase_counts = Counter(
            sample["phase"]
            for sample in self.samples
        )

        missing_phases = [
            phase
            for phase, target
            in PHASE_TARGETS.items()
            if target > 0
            and self.phase_counts.get(
                phase,
                0,
            ) == 0
        ]

        if missing_phases:
            raise RuntimeError(
                f"Missing target phases: "
                f"{missing_phases}"
            )

        probability_sum = sum(
            PHASE_TARGETS.values()
        )

        if not np.isclose(
            probability_sum,
            1.0,
        ):
            raise RuntimeError(
                f"PHASE_TARGETS must sum to 1.0, "
                f"got {probability_sum}"
            )

        print(
            f"[*] Loaded "
            f"{len(self.samples)} transitions "
            f"across {len(npz_paths)} episode(s)."
        )

        print(
            "[*] Natural temporal phase distribution:"
        )

        for phase in PHASE_TARGETS:
            count = self.phase_counts.get(
                phase,
                0,
            )

            percentage = (
                100.0
                * count
                / len(self.samples)
            )

            print(
                f"    {phase:24s} "
                f"{count:3d} frames "
                f"({percentage:5.2f}%)"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = sample[
            "image"
        ].transpose(
            2,
            0,
            1,
        ).copy()

        image = (
            torch.from_numpy(
                image
            ).float()
            / 255.0
        ) * 2.0 - 1.0

        return {
            "image": image,
            "state": torch.from_numpy(
                sample["state"]
            ).float(),
            "action_chunk": torch.from_numpy(
                sample["action_chunk"]
            ).float(),
            "frame": sample["frame"],
            "phase": sample["phase"],
        }

    def build_sampler(
        self,
        samples_per_epoch: int,
        seed: int,
    ) -> WeightedRandomSampler:
        if samples_per_epoch < 1:
            raise ValueError(
                "samples_per_epoch must be >= 1."
            )

        weights = [
            (
                PHASE_TARGETS[
                    sample["phase"]
                ]
                / self.phase_counts[
                    sample["phase"]
                ]
            )
            for sample in self.samples
        ]

        generator = torch.Generator()
        generator.manual_seed(seed)

        return WeightedRandomSampler(
            weights=torch.as_tensor(
                weights,
                dtype=torch.double,
            ),
            num_samples=samples_per_epoch,
            replacement=True,
            generator=generator,
        )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def initialize_smolvla_training_policy(
    device: torch.device,
) -> Tuple[SmolVLAPolicy, SmolVLAConfig]:
    print(
        f"[*] Initializing SmolVLA "
        f"on {device}..."
    )

    config_path = hf_hub_download(
        repo_id="lerobot/smolvla_base",
        filename="config.json",
    )

    with open(
        config_path,
        encoding="utf-8",
    ) as file:
        config_dict = json.load(file)

    config_dict.pop(
        "type",
        None,
    )

    config_dict[
        "load_vlm_weights"
    ] = False

    config_dict[
        "freeze_vision_encoder"
    ] = True

    config_dict[
        "train_expert_only"
    ] = True

    config_dict[
        "train_state_proj"
    ] = True

    config_dict[
        "device"
    ] = str(device)

    # V5 H=20 configuration
    config_dict[
        "chunk_size"
    ] = CHUNK_SIZE

    config_dict[
        "n_action_steps"
    ] = CHUNK_SIZE

    config_dict[
        "input_features"
    ] = {
        key: PolicyFeature(
            type=FeatureType[
                value["type"]
            ],
            shape=tuple(
                value["shape"]
            ),
        )
        for key, value
        in config_dict[
            "input_features"
        ].items()
    }

    config_dict[
        "output_features"
    ] = {
        key: PolicyFeature(
            type=FeatureType[
                value["type"]
            ],
            shape=tuple(
                value["shape"]
            ),
        )
        for key, value
        in config_dict[
            "output_features"
        ].items()
    }

    config_dict[
        "normalization_mapping"
    ] = {
        key: NormalizationMode[
            value
        ]
        for key, value
        in config_dict[
            "normalization_mapping"
        ].items()
    }

    config = SmolVLAConfig(
        **config_dict
    )

    policy = SmolVLAPolicy(
        config
    )

    weights_path = hf_hub_download(
        repo_id="lerobot/smolvla_base",
        filename="model.safetensors",
    )

    policy.load_state_dict(
        load_file(
            weights_path
        ),
        strict=False,
    )

    policy.model.vlm_with_expert.freeze_vision_encoder = True
    policy.model.vlm_with_expert.train_expert_only = True

    policy.model.vlm_with_expert.set_requires_grad()
    policy.model.set_requires_grad()

    policy.to(device)

    trainable_parameters = [
        parameter
        for parameter
        in policy.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    print(
        f"[+] Loaded model. "
        f"Trainable parameters: "
        f"{trainable_count / 1e6:.2f} M."
    )

    return policy, config


def save_checkpoint(
    policy: SmolVLAPolicy,
    checkpoint_path: Path,
):
    state_dict = {
        key: value.detach().contiguous().cpu()
        for key, value
        in policy.state_dict().items()
    }

    save_file(
        state_dict,
        str(checkpoint_path),
    )


def train_phase_balanced_v5(
    dataset_dir: str = (
        "datasets/lerobot_dataset/"
        "glass_pick_place_iteration11"
    ),
    expert_replay: str = (
        "datasets/cotracker_iterations/"
        "iteration_11_full_mujoco/"
        "cup_lifting_01_replay.npz"
    ),
    out_dir: str = (
        "outputs/"
        "smolvla_glass_phase_balanced_v5"
    ),
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 6,
):
    if epochs < 1:
        raise ValueError(
            "epochs must be >= 1."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1."
        )

    if lr <= 0:
        raise ValueError(
            "lr must be positive."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    device = select_device()

    output_path = Path(
        out_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if device.type == "cuda":
        print(
            f"[+] CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 80)
    print(
        "Phase-Balanced SmolVLA V5 "
        "Temporal-Sequence H=20 Fine-Tuning"
    )

    print(
        f"Device: {device} | "
        f"Epochs: {epochs} | "
        f"Batch Size: {batch_size} | "
        f"LR: {lr}"
    )

    print("=" * 80)

    episode_path = (
        Path(dataset_dir)
        / "cup_lifting_01_episode.npz"
    )

    train_dataset = (
        PhaseBalancedGlassDataset(
            npz_paths=[
                str(
                    episode_path
                )
            ],
            expert_replay_path=(
                expert_replay
            ),
            chunk_size=CHUNK_SIZE,
            max_action_dim=(
                MAX_ACTION_DIM
            ),
        )
    )

    samples_per_epoch = len(
        train_dataset
    )

    sampler = (
        train_dataset.build_sampler(
            samples_per_epoch=(
                samples_per_epoch
            ),
            seed=seed,
        )
    )

    expected_samples = {
        phase: round(
            probability
            * samples_per_epoch
        )
        for phase, probability
        in PHASE_TARGETS.items()
    }

    print(
        "[*] Expected samples per epoch:"
    )

    for phase in PHASE_TARGETS:
        print(
            f"    {phase:24s} "
            f"~{expected_samples[phase]:3d} "
            f"({PHASE_TARGETS[phase] * 100:5.1f}%)"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    policy, _ = (
        initialize_smolvla_training_policy(
            device
        )
    )

    print(
        f"[*] Dataset chunk size: "
        f"{train_dataset.chunk_size}"
    )

    print(
        f"[*] Model chunk size:   "
        f"{policy.config.chunk_size}"
    )

    print(
        f"[*] Model action steps: "
        f"{policy.config.n_action_steps}"
    )

    if (
        train_dataset.chunk_size
        != CHUNK_SIZE
    ):
        raise RuntimeError(
            f"Dataset chunk mismatch: "
            f"{train_dataset.chunk_size} "
            f"!= {CHUNK_SIZE}"
        )

    if (
        policy.config.chunk_size
        != CHUNK_SIZE
    ):
        raise RuntimeError(
            f"Model chunk mismatch: "
            f"{policy.config.chunk_size} "
            f"!= {CHUNK_SIZE}"
        )

    if (
        policy.config.n_action_steps
        != CHUNK_SIZE
    ):
        raise RuntimeError(
            f"Model action-step mismatch: "
            f"{policy.config.n_action_steps} "
            f"!= {CHUNK_SIZE}"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            "HuggingFaceTB/"
            "SmolVLM2-500M-Video-Instruct"
        )
    )

    tokens = tokenizer(
        [
            DEFAULT_TASK_PROMPT
            + "\n"
        ],
        padding="max_length",
        truncation=True,
        max_length=(
            policy.config
            .tokenizer_max_length
        ),
        return_tensors="pt",
    )

    language_tokens_single = (
        tokens[
            "input_ids"
        ].to(device)
    )

    language_masks_single = (
        tokens[
            "attention_mask"
        ].to(
            device,
            dtype=torch.bool,
        )
    )

    print(
        f"[+] Language tokens: "
        f"{tuple(language_tokens_single.shape)}"
    )

    trainable_parameters = [
        parameter
        for parameter
        in policy.parameters()
        if parameter.requires_grad
    ]

    optimizer = (
        torch.optim.AdamW(
            trainable_parameters,
            lr=lr,
            weight_decay=1e-4,
        )
    )

    total_steps = (
        len(train_loader)
        * epochs
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )
    )

    best_train_loss = float(
        "inf"
    )

    history = {
        "train_loss": [],
        "epoch_times": [],
        "sampled_phase_counts": [],
    }

    training_start = (
        time.time()
    )

    print(
        f"\n[*] Starting training loop "
        f"({epochs} epochs, "
        f"{len(train_loader)} "
        f"batches/epoch)..."
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start = (
            time.time()
        )

        policy.train()

        epoch_losses = []

        sampled_phase_counts = (
            Counter()
        )

        for batch_idx, batch in enumerate(
            train_loader
        ):
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            states = batch[
                "state"
            ].to(
                device,
                non_blocking=True,
            )

            action_chunks = batch[
                "action_chunk"
            ].to(
                device,
                non_blocking=True,
            )

            if (
                epoch == 1
                and batch_idx == 0
            ):
                print(
                    f"[*] Training action "
                    f"chunk shape: "
                    f"{tuple(action_chunks.shape)}"
                )

            if (
                action_chunks.shape[1]
                != CHUNK_SIZE
            ):
                raise RuntimeError(
                    f"Training action "
                    f"chunk horizon mismatch: "
                    f"{action_chunks.shape}"
                )

            sampled_phase_counts.update(
                batch["phase"]
            )

            current_batch_size = (
                images.shape[0]
            )

            image_masks = torch.ones(
                current_batch_size,
                dtype=torch.bool,
                device=device,
            )

            padded_states = (
                pad_vector(
                    states,
                    policy.config
                    .max_state_dim,
                )
            )

            language_tokens = (
                language_tokens_single
                .expand(
                    current_batch_size,
                    -1,
                )
            )

            language_masks = (
                language_masks_single
                .expand(
                    current_batch_size,
                    -1,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss_tensor = (
                policy.model(
                    [images],
                    [image_masks],
                    language_tokens,
                    language_masks,
                    padded_states,
                    action_chunks,
                )
            )

            loss = (
                loss_tensor.mean()
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"Non-finite loss "
                    f"at epoch {epoch}, "
                    f"batch {batch_idx}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        average_train_loss = (
            float(
                np.mean(
                    epoch_losses
                )
            )
        )

        epoch_duration = (
            time.time()
            - epoch_start
        )

        history[
            "train_loss"
        ].append(
            average_train_loss
        )

        history[
            "epoch_times"
        ].append(
            epoch_duration
        )

        history[
            "sampled_phase_counts"
        ].append(
            dict(
                sampled_phase_counts
            )
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
        ):
            current_lr = (
                optimizer
                .param_groups[0][
                    "lr"
                ]
            )

            print(
                f"[Epoch "
                f"{epoch:02d}/"
                f"{epochs:02d}] "
                f"Train Loss: "
                f"{average_train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: "
                f"{epoch_duration:.1f}s"
            )

            print(
                "    Sampled phases: "
                + ", ".join(
                    f"{phase}="
                    f"{sampled_phase_counts.get(phase, 0)}"
                    for phase
                    in PHASE_TARGETS
                )
            )

        if (
            average_train_loss
            < best_train_loss
        ):
            best_train_loss = (
                average_train_loss
            )

            save_checkpoint(
                policy,
                output_path
                / "model.safetensors",
            )

        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    total_training_time = (
        time.time()
        - training_start
    )

    initial_loss = (
        history[
            "train_loss"
        ][0]
    )

    final_loss = (
        history[
            "train_loss"
        ][-1]
    )

    loss_drop = (
        100.0
        * (
            1.0
            - final_loss
            / initial_loss
        )
        if initial_loss > 0
        else 0.0
    )

    print(
        f"\n[+] Training completed "
        f"in {total_training_time:.1f}s."
    )

    print(
        f"[+] Initial Train Loss: "
        f"{initial_loss:.4f} -> "
        f"Final: {final_loss:.4f} "
        f"({loss_drop:.1f}% drop)"
    )

    print(
        f"[+] Best Training Loss: "
        f"{best_train_loss:.4f}"
    )

    config_output = {
        "model_type": (
            "smolvla_glass_phase_balanced_v5"
        ),
        "base_model": (
            "lerobot/smolvla_base"
        ),
        "dataset": str(
            episode_path
        ),
        "expert_replay": (
            expert_replay
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": CHUNK_SIZE,
        "state_dim": 8,
        "action_dim": 7,
        "max_action_dim": (
            MAX_ACTION_DIM
        ),
        "sampling_mode": (
            "temporal_sequence_"
            "phase_balanced_v5_h20"
        ),
        "samples_per_epoch": (
            samples_per_epoch
        ),
        "natural_phase_counts": dict(
            train_dataset.phase_counts
        ),
        "target_phase_probabilities": (
            PHASE_TARGETS
        ),
        "expert_boundaries": (
            train_dataset.boundaries
        ),
        "release_settle_frames": (
            RELEASE_SETTLE_FRAMES
        ),
        "action_normalization": (
            "none"
        ),
        "validation_mode": (
            "none - one expert episode"
        ),
        "initial_train_loss": (
            initial_loss
        ),
        "final_train_loss": (
            final_loss
        ),
        "best_train_loss": (
            best_train_loss
        ),
        "total_time_s": (
            total_training_time
        ),
        "prompt": (
            DEFAULT_TASK_PROMPT
        ),
    }

    with open(
        output_path
        / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_output,
            file,
            indent=2,
        )

    with open(
        output_path
        / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

    print(
        f"[+] Saved V5 checkpoint "
        f"and metadata: "
        f"{output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase-balanced SmolVLA V5 "
            "temporal-sequence H=20 "
            "fine-tuning"
        )
    )

    parser.add_argument(
        "--dataset_dir",
        default=(
            "datasets/lerobot_dataset/"
            "glass_pick_place_iteration11"
        ),
    )

    parser.add_argument(
        "--expert_replay",
        default=(
            "datasets/cotracker_iterations/"
            "iteration_11_full_mujoco/"
            "cup_lifting_01_replay.npz"
        ),
    )

    parser.add_argument(
        "--out_dir",
        default=(
            "outputs/"
            "smolvla_glass_phase_balanced_v5"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    train_phase_balanced_v5(
        dataset_dir=(
            args.dataset_dir
        ),
        expert_replay=(
            args.expert_replay
        ),
        out_dir=(
            args.out_dir
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


"""
Save as:
    src/learning/train_smolvla_phase_balanced_v5.py


Syntax check:

cd ~/Documents/Pollex && \
python3 -m py_compile src/learning/train_smolvla_phase_balanced_v5.py


Smoke test:

cd ~/Documents/Pollex && \
PYTHONPATH=. python3 -m src.learning.train_smolvla_phase_balanced_v5 \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --out_dir outputs/smolvla_glass_phase_balanced_v5_smoke \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 6


The smoke test MUST report:

    Dataset chunk size: 20
    Model chunk size:   20
    Model action steps: 20
    Training action chunk shape: (4, 20, 32)


Full training:

cd ~/Documents/Pollex && \
rm -rf outputs/smolvla_glass_phase_balanced_v5 && \
PYTHONPATH=. python3 -m src.learning.train_smolvla_phase_balanced_v5 \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --out_dir outputs/smolvla_glass_phase_balanced_v5 \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 6
"""