# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch GPT2/3 Dataloader."""

import logging
import torch

from cerebras.modelzoo.data.common.HDF5IterableDataProcessor import (
    HDF5IterableDataProcessor,
)
from cerebras.modelzoo.data.nlp.gpt.config import GptHDF5DataProcessorConfig

from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field, PositiveInt, field_validator, model_validator
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.data.common.input_utils import is_distributed


class GptHDF5DataProcessor(HDF5IterableDataProcessor):
    """
    A HDF5 dataset processor for GPT pre-training.
    Loads data from HDF5 files.

    Args:
        config: The configuration object for the GPT HDF5 data processor.
    """

    def __init__(self, config: GptHDF5DataProcessorConfig):
        if isinstance(config, dict):
            config = GptHDF5DataProcessorConfig(**config)

        if config.max_sequence_length is not None:
            logging.warning(
                "`max_sequence_length` is not used in for GptHDF5DataProcessor "
                "as it expects the data to be pre tokenized to a desired MSL, "
                "please remove it from the supplied config."
            )

        # The super class will take care of sharding the dataset and creating the dataloader
        super().__init__(config)



class GptInferenceDataset(Dataset):
    """
    A class representing a GPTInferenceDataset inheriting torch.utils.data.Dataset.
    """

    def __init__(self, data, data_processor):
        self.data = data
        self.length = data_processor.num_examples * data_processor.n_responses
        super(GptInferenceDataset, self).__init__()

    def __getitem__(self, index):
        feature = {
            "input_ids": self.data["input_ids"][index],
            "stop_sequences": self.data["stop_sequences"][index],
            "rand_uniform": self.data["rand_uniform"][index],
            "context_length": self.data["ctx_len"][index],
        }

        return feature

    def __len__(self):
        return self.length


class GptInferenceSyntheticDataProcessorConfig(DataConfig):
    data_processor: Literal["GptInferenceSyntheticDataProcessor"]

    num_examples: int = ...
    vocab_size: Optional[int] = None
    max_sequence_length: Optional[int] = None
    n_responses: PositiveInt = ...
    shuffle: bool = ...
    shuffle_seed: Optional[int] = None
    batch_size: PositiveInt = ...

    batch_sampler: Optional[torch.utils.data.sampler.Sampler] = None
    pin_memory: bool = False
    timeout: int = 0
    sampler: Optional[torch.utils.data.sampler.Sampler] = None

    num_workers: int = 8
    prefetch_factor: Optional[Any] = Field(None, deprecated=True)
    persistent_workers: Optional[Any] = Field(None, deprecated=True)
    drop_last: bool = True
    start_token: int = 111  # Default start token for GPT-2
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """


class GptInferenceSyntheticDataProcessor:
    """
    Synthetic dataset generator.
    Args:
        config: The configuration object used
    """

    def __init__(self, config: GptInferenceSyntheticDataProcessorConfig):
        if isinstance(config, dict):
            config = GptInferenceSyntheticDataProcessorConfig(**config)

        self.num_examples = config.num_examples
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_sequence_length

        # batch size should be multiple of n_responses
        self.n_responses = config.n_responses

        # Check more about streaming batch size
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.batch_size = self.batch_size * self.n_responses
        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed

        self.sampler = config.sampler
        self.batch_sampler = config.batch_sampler
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.drop_last = config.drop_last
        self.timeout = config.timeout

        self.start_token = config.start_token
        self.collate_fn = default_collate

    def create_dataloader(self):
        """
        Create dataloader.
        :returns: dataloader
        """
        np.random.seed(seed=1)
        data = dict()

        # Generate base random data for num_examples
        base_data = np.random.randint(
            low=0,
            high=self.vocab_size - 1,
            size=(self.num_examples, self.max_seq_len),
            dtype=np.int32,
        )

        rand_id = np.random.randint(1, self.max_seq_len)
        base_data[:, -rand_id:] = self.start_token

        # Repeat each row n_responses times
        data["input_ids"] = np.repeat(base_data, self.n_responses, axis=0)

        # Boolean mask where token matches
        mask = (
            data["input_ids"] == self.start_token
        )  # shape: (batch_size, seq_len)
        data["ctx_len"] = mask.argmax(axis=1)  # shape: (batch_size,)

        data["stop_sequences"] = [
            np.array([50256]).astype(np.int32)
            for _ in range(self.num_examples * self.n_responses)
        ]
        # data["stop_sequences"] = [
        #    [] for _ in range(self.num_examples * self.n_responses)
        # ]

        data["rand_uniform"] = [
            np.array([np.random.uniform()]).astype(np.float32)
            for _ in range(self.num_examples * self.n_responses)
        ]

        dataset = GptInferenceDataset(data, self)

        if is_distributed():
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=self.shuffle,
                seed=self.shuffle_seed,
            )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
        )
