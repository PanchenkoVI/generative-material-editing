import os
import lightning as L 
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
 
wandb = None

class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config
        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )
        if not self.use_wandb:
            self.writer = SummaryWriter(log_dir=f"{self.save_path}/{self.run_name}/logs")
        else:
            self.writer = None
        self.to_tensor = T.ToTensor()

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)
        else:
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            self.writer.add_scalar('loss/train', loss_value, self.total_steps)
            self.writer.add_scalar('t', pl_module.last_t, self.total_steps)
            self.writer.add_scalar('gradient_size', gradient_size, self.total_steps)
            self.writer.add_scalar('epoch', trainer.current_epoch, self.total_steps)
            self.writer.add_scalar('loss_sd', pl_module.res['loss_sd'].item(), self.total_steps)
            self.writer.add_scalar('loss_sd', pl_module.res['loss'].item(), self.total_steps)
            self.writer.add_scalar('loss_mask', pl_module.res['loss_mask'].item(), self.total_steps)
            if 'loss_odm' in pl_module.res:
                self.writer.add_scalar('loss_odm', pl_module.res['loss_odm'].item(), self.total_steps)
            if 'loss_ocr' in pl_module.res:
                self.writer.add_scalar('loss_ocr', pl_module.res['loss_ocr'].item(), self.total_steps)
                self.writer.add_scalar('loss_ctc', pl_module.res['loss_ctc'].item(), self.total_steps)


        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            ) 