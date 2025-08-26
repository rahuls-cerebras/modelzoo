from typing import List, Literal, Optional, Union

import numpy as np
import torch
from pydantic import PositiveInt

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.common.input_utils import is_distributed


class NpyRLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
    ):
        super().__init__()
        rollouts_path = data_dir + "/rollouts.npz"
        ref_rollouts_path = data_dir + "/ref_rollouts.npz"
        advantages_path = data_dir + "/advantages.npz"
        self.advantages = None
        self.ref_log_probs = None
        self.full_response = None
        self.msl = None
        self.input_len = None
        self.response_len = None
        self.attention_mask = None

        try:
            with open(rollouts_path, 'rb') as f:
                samples = np.load(f)
                self._dataset_size = len(samples['first'])
                self.full_response = samples['first'].astype(np.int32)
                self.input_len = samples['second']
                self.response_len = samples['third']
                self.msl = samples['fourth']
        except Exception as e:
            raise RuntimeError(f"Failed to read : {rollouts_path}") from e
        try:
            with open(advantages_path, 'rb') as f:
                samples = np.load(f)
                self.advantages = samples['first']
        except Exception as e:
            raise RuntimeError(f"Failed to read : {advantages_path}") from e

        import os

        if os.path.exists(ref_rollouts_path):
            try:
                with open(ref_rollouts_path, 'rb') as f:
                    samples = np.load(f)
                    self.ref_log_probs = samples['first']
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read : {ref_rollouts_path}"
                ) from e

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, idx):
        input_len = self.input_len[idx]
        response_len = self.response_len[idx]
        # Initialize all zeros
        attention_mask = np.zeros(self.msl, dtype=np.int32)
        # Fill the first (input_len + response_len) positions with 1
        total_len = input_len + response_len
        attention_mask[:total_len] = 1

        if self.ref_log_probs is not None:
            advantages = np.zeros(self.msl, dtype=np.float32)
            advantages[input_len : input_len + response_len] = self.advantages[
                idx
            ]
            loss_mask = np.zeros(self.msl, dtype=np.float32)
            loss_mask[input_len : input_len + response_len] = 1.0

            data = {
                "input_ids": self.full_response[idx],
                "attention_mask": attention_mask,
                "advantages": advantages,
                "loss_mask": loss_mask,
                "ref_log_probs": self.ref_log_probs[idx],
            }
        else:
            data = {
                "input_ids": self.full_response[idx],
                "attention_mask": attention_mask,
                "labels": self.full_response[idx],
            }
        return data


class GptNpyRLDataProcessorConfig(DataConfig):
    data_processor: Literal["GptNpyRLDataProcessor"]

    num_workers: int = 0
    """ The number of PyTorch processes used in the dataloader. """

    prefetch_factor: Optional[int] = 10
    """ The number of batches to prefetch in the dataloader. """

    persistent_workers: bool = True

    batch_size: PositiveInt = ...

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    "Path to the data files to use."

    sampler: Optional[torch.utils.data.sampler.Sampler] = None


class GptNpyRLDataProcessor:
    """
    A map style dataset for GPT style models.

    Supports data saved on disk in either of the following formats:
        - `(num_tokens,)`, i.e. a set of documents tokenized and concatenated.
            We refer to this as the 'corpus' format in what follows.
        - `(num_sequences, 3, sequence_length)`, i.e. data that has already
            been preprocessed into sequences. We refer to this as the
            'sample' format in what follows.

    Args:
        config: The config used to configure the data processor.
    """

    def __init__(self, config: GptNpyRLDataProcessorConfig):
        if isinstance(config, dict):
            config = GptNpyRLDataProcessorConfig(**config)

        self.config = config

        self.dataset = NpyRLDataset(config.data_dir)
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.sampler = config.sampler

        if is_distributed():
            assert self.sampler is None, "Cannot use sampler in config with DDP"
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=False,
                seed=1,
            )

    def create_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            batch_sampler=self.sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
        )
