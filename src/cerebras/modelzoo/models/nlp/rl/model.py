from copy import deepcopy
from typing import Literal

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.losses.GRPOLoss import GRPOLoss
from cerebras.modelzoo.models.nlp.gpt2.model import Gpt2Model, GPT2ModelConfig


class RLModelConfig(GPT2ModelConfig):
    name: Literal["RL"]


class RLModel(torch.nn.Module):
    def __init__(self, config: RLModelConfig):
        super().__init__()

        self.policy_model = Gpt2Model(config)

        self.ref_model = deepcopy(self.policy_model)
        self.ref_model.eval()

        self.loss_fn = GRPOLoss()

    def forward(self, data):
        with torch.no_grad():
            _, old_log_probs = self.ref_model(
                data={
                    "input_ids": data["input_ids"],
                    "attention_mask": data["attention_mask"],
                    "labels": data["input_ids"],
                },
                output_logits=True,
            )
            old_log_probs = torch.log_softmax(old_log_probs, dim=-1)
            old_log_probs = torch.gather(
                input=old_log_probs,
                dim=-1,
                index=data["input_ids"].to(torch.int64).unsqueeze(-1),
            ).squeeze(-1)

        _, curr_log_probs = self.policy_model(
            data={
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "labels": data["input_ids"],
            },
            output_logits=True,
        )

        curr_log_probs = torch.log_softmax(curr_log_probs, dim=-1)
        # curr_log_probs = torch.gather(
        #    input=curr_log_probs,
        #    dim=-1,
        #    index=data["input_ids"].to(torch.int64).unsqueeze(-1),
        # ).squeeze(-1)
        one_hot = cstorch.nn.functional.one_hot(
            data["input_ids"].to(torch.int64),
            num_classes=curr_log_probs.size(-1),
        )  # .to(curr_log.probs.dtype)  # (8,128,50257)
        # Multiply and sum over vocab dimension
        curr_log_probs = torch.sum(curr_log_probs * one_hot, dim=-1)  # (8,128)
        loss = self.loss_fn(
            old_log_probs,
            curr_log_probs,
            data["advantages"],
            data["loss_mask"],
            data["ref_log_probs"],
        )
        return loss
