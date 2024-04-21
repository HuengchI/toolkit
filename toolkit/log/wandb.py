from transformers import TrainerCallback

class HfWandbCallback(TrainerCallback):
    def __init__(self, wandb_runner, **kwargs) -> None:
        self.wandb_runner = wandb_runner
        super().__init__()
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        # main process log
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self.wandb_runner.summary[k] = v # Log Summary Metrics
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            # non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})
