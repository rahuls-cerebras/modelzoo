import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback


class Inference(Callback):
    """
    Callback class to post-process model output logits to generate continuation
    strings until a specified token is generated.
    """

    def __init__(self, start_token):
        """
        Args:
            tokenizer: Tokenizer object used to decode the generated continuation
            metadata: List of tuples of (stop token sequences, ctx length)
                for each sample in the batch.
            gen_kwargs: Dict specifying settings for generative inference.
        """
        self.start_token = start_token
        self.results = []
        self.input_len = []
        self.response_len = []

    def on_before_forward(self, trainer, model, batch, args, kwargs):
        kwargs["autoregressive"] = True

    def on_after_forward(self, trainer, model, outputs, batch):
        self.post_process(
            predictions=outputs["output"], ctx_len=batch["context_length"]
        )

    @cstorch.step_closure
    def post_process(self, predictions, ctx_len):
        """
        Post-processes the model output logits to generate continuation strings.

        Args:
            predictions: Tensor of shape (batch_size, max_seq_len)
                containing the model's predictions
        """
        # Post processing of model output to produce results
        for i, pred in enumerate(predictions):
            response = pred[ctx_len[i] :].tolist()
            if self.start_token in response:
                self.response_len.append(
                    response.index(self.start_token) - ctx_len[i]
                )
            else:
                self.response_len.append(len(pred) - ctx_len[i])
            self.results.append(pred)
            self.input_len.append(ctx_len[i])

        import numpy as np

        outs = np.array(self.results)
        # Save to .npz file
        np.savez(
            "rollouts.npz",
            first=np.array(self.results),
            second=np.array(self.input_len),
            third=np.array(self.response_len),
            fourth=len(predictions[0]),
        )

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
