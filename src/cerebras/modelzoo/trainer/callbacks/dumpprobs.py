import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback


class DumpProbs(Callback):
    """
    Callback class to post-process model output logits to generate continuation
    strings until a specified token is generated.
    """

    def __init__(self):
        """
        Args:
            tokenizer: Tokenizer object used to decode the generated continuation
            metadata: List of tuples of (stop token sequences, ctx length)
                for each sample in the batch.
            gen_kwargs: Dict specifying settings for generative inference.
        """
        self.results = []

    def on_before_forward(self, trainer, model, batch, args, kwargs):
        kwargs["output_logits"] = True

    def on_after_forward(self, trainer, model, outputs, batch):
        import torch

        self.post_process(
            predictions=outputs["logits"].to(torch.float32),
            attention_mask=batch["attention_mask"],
            input_ids=batch["input_ids"],
        )

    @cstorch.step_closure
    def post_process(self, predictions, input_ids, attention_mask):
        import torch

        """
        Post-processes the model output logits to generate continuation strings.

        Args:
            predictions: Tensor of shape (batch_size, max_seq_len)
                containing the model's predictions
        """
        # Post processing of model output to produce results
        for i, pred in enumerate(predictions):
            # Get tokens for the generated continuation string
            # gen_continuation = pred[ctx_len[i]:].tolist()
            log_probs = torch.log_softmax(pred, dim=-1)
            selected_log_probs = torch.gather(
                input=log_probs,
                dim=-1,
                index=input_ids[i].to(torch.int64).unsqueeze(-1),
            ).squeeze(-1)
            selected_log_probs = (
                selected_log_probs * attention_mask[i]
            )  # Zero out where mask is 0
            self.results.append(selected_log_probs)

        import numpy as np

        outs = np.array(self.results)
        # Save to .npz file
        np.savez("ref_rollouts.npz", first=outs)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
