"""Training utilities for FunTorch5 with robust monitoring."""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


OptimizerConfig = Mapping[str, Any]
SchedulerConfig = Optional[Mapping[str, Any]]
TrainingConfig = Mapping[str, Any]


@dataclass
class EpochMetrics:
    loss: float
    correlation: Optional[float] = None
    output_variance: Optional[float] = None
    abs_error: Optional[float] = None
    duration_sec: float = 0.0
    accuracy: Optional[float] = None


@dataclass
class TrainingHistory:
    train: List[EpochMetrics] = field(default_factory=list)
    val: List[EpochMetrics] = field(default_factory=list)


class _NoOpGradScaler:
    def __init__(self) -> None:
        self._enabled = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def is_enabled(self) -> bool:
        return self._enabled

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        return None


class Trainer:
    """Wrapper around a PyTorch model that enforces monitoring & checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: OptimizerConfig,
        training_cfg: TrainingConfig,
        scheduler_cfg: SchedulerConfig = None,
        checkpoint_dir: str | Path = "models",
        resume_path: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.training_cfg = dict(training_cfg)
        self.optimizer_cfg = dict(optimizer_cfg)
        self.scheduler_cfg = dict(scheduler_cfg or {})
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        device_str = self.training_cfg.get("device")
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        loss_cfg = self.training_cfg.get("loss_config", {})
        loss_type = str(loss_cfg.get("type", loss_cfg.get("mode", ""))).lower()
        use_bce = bool(loss_cfg.get("use_bce", False))
        self.is_classification = use_bce or loss_type in {"bce", "binary", "classification"}
        if self.is_classification:
            pos_weight = loss_cfg.get("pos_weight")
            weight_tensor = (
                torch.tensor(float(pos_weight), device=self.device) if pos_weight is not None else None
            )
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
            self.corr_weight = 0.0
            self.corr_epsilon = 0.0
        else:
            if loss_cfg.get("use_huber_loss", True):
                delta = float(loss_cfg.get("huber_delta", 0.05))
                self.criterion = nn.HuberLoss(delta=delta)
            else:
                self.criterion = nn.MSELoss()
            self.corr_weight = float(loss_cfg.get("correlation_weight", 0.0))
            self.corr_epsilon = float(loss_cfg.get("correlation_epsilon", 1e-6))

        self.grad_accum_steps = int(self.training_cfg.get("grad_accum_steps", 1))
        self.max_epochs = int(self.training_cfg.get("num_epochs", 1))
        self.early_stop_patience = int(
            self.training_cfg.get("early_stopping", {}).get("early_stop_patience", 20)
        )
        self.clip_grad_norm = self.training_cfg.get("clip_grad_norm")
        self.device_type = self.device.type
        amp_requested = bool(self.training_cfg.get("amp", True))
        if self.device_type == "cuda" and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp_requested)
            self._autocast = lambda enabled: torch.amp.autocast("cuda", enabled=enabled)
        else:
            self.scaler = _NoOpGradScaler()
            self._autocast = lambda enabled: contextlib.nullcontext()
        self.history = TrainingHistory()
        self.plot_interval = int(self.training_cfg.get("plot_interval", 10))

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.start_epoch = 0
        self.best_val_loss = math.inf
        self.not_improved_epochs = 0

        if resume_path:
            self.load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.optimizer_cfg.get("name", "adamw").lower()
        lr = float(self.optimizer_cfg.get("lr", 1e-3))
        weight_decay = float(self.optimizer_cfg.get("weight_decay", 0.0))
        params = self.model.parameters()
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            momentum = float(self.optimizer_cfg.get("momentum", 0.9))
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        raise ValueError(f"Unsupported optimizer: {name}")

    # ------------------------------------------------------------------
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if not self.scheduler_cfg:
            return None
        sched_type = self.scheduler_cfg.get("scheduler_type", "plateau").lower()
        if sched_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(self.scheduler_cfg.get("lr_factor", 0.5)),
                patience=int(self.scheduler_cfg.get("lr_patience", 5)),
            )
        if sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(self.scheduler_cfg.get("t_max", 10))
            )
        if sched_type in {"cosine_restart", "cosine_warm_restart", "cosinewarmrestart"}:
            t_mult_cfg = self.scheduler_cfg.get("t_mult", 1)
            t_mult = int(round(float(t_mult_cfg)))
            if t_mult < 1:
                raise ValueError(f"cosine warm restart requires t_mult >= 1, got {t_mult_cfg}")
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(self.scheduler_cfg.get("t_0", self.scheduler_cfg.get("t_max", 10))),
                T_mult=t_mult,
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
        if sched_type == "onecycle":
            max_lr = float(self.scheduler_cfg.get("max_lr", self.optimizer_cfg.get("lr", 1e-3)))
            steps_per_epoch = int(self.scheduler_cfg.get("steps_per_epoch"))
            epochs = int(self.scheduler_cfg.get("epochs", self.max_epochs))
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=float(self.scheduler_cfg.get("pct_start", 0.3)),
            )
        raise ValueError(f"Unsupported scheduler_type: {sched_type}")

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        epoch_callback: Optional[Callable[[int, EpochMetrics, Optional[EpochMetrics]], None]] = None,
    ) -> TrainingHistory:
        total_epochs = epochs or self.max_epochs
        for epoch in range(self.start_epoch, total_epochs):
            print(f"Epoch {epoch + 1}/{total_epochs}")
            train_metrics = self.train_one_epoch(train_loader, epoch)
            self.history.train.append(train_metrics)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                self.history.val.append(val_metrics)
                val_loss = val_metrics.loss
                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    self.not_improved_epochs = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.not_improved_epochs += 1
                    if self.not_improved_epochs >= self.early_stop_patience:
                        print("Early stopping triggered.")
                        break
            else:
                improved = True

            if not val_loader:
                if (epoch + 1) % self.training_cfg.get("checkpoint_interval", 5) == 0:
                    self.save_checkpoint(epoch, is_best=False)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_metrics is not None:
                    self.scheduler.step(val_metrics.loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            self.save_checkpoint(epoch, is_best=improved and val_loader is not None, latest=True)
            self._log_epoch_summary(epoch, train_metrics, val_metrics)
            if epoch_callback is not None:
                try:
                    epoch_callback(epoch, train_metrics, val_metrics)
                except Exception as exc:  # pragma: no cover - defensive logging hook
                    print(f"[Warning] epoch_callback raised {exc!r}; continuing training.")
            if self.plot_interval > 0 and (
                (epoch + 1) % self.plot_interval == 0 or epoch + 1 == total_epochs
            ):
                self._save_metric_plots(epoch + 1)
                snapshot_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
                self.save_checkpoint(epoch, is_best=False, latest=False, extra_path=snapshot_path)
        return self.history

    # ------------------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        self.model.train()
        running_loss = 0.0
        sample_count = 0
        start_time = time.time()
        progress = tqdm(loader, desc="Train", leave=False)
        output_vars: List[float] = []
        correct = 0
        total = 0
        for step, (inputs, targets) in enumerate(progress):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with self._autocast(self.scaler.is_enabled()):
                outputs = self.model(inputs)
                outputs = outputs.float()
                loss = self.criterion(outputs, targets)
                if self.corr_weight > 0.0:
                    corr_penalty = self._correlation_penalty(outputs, targets)
                    loss = loss + self.corr_weight * corr_penalty
            self.scaler.scale(loss / self.grad_accum_steps).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)
            variance = torch.var(outputs).item()
            output_vars.append(variance)
            if self.is_classification:
                probs = torch.sigmoid(outputs.detach())
                preds_bin = (probs >= 0.5).float()
                correct += (preds_bin == targets).sum().item()
                total += targets.numel()
            if variance < 1e-4:
                status: Dict[str, float | str] = {"warning": "mode collapse suspected"}
            else:
                status = {"loss": loss.item(), "var": variance}
            if self.is_classification and total > 0:
                status["acc"] = correct / total
            progress.set_postfix(status)
        progress.close()
        avg_loss = running_loss / max(sample_count, 1)
        avg_var = float(np.mean(output_vars)) if output_vars else None
        duration = time.time() - start_time
        train_accuracy = correct / total if self.is_classification and total > 0 else None
        return EpochMetrics(
            loss=avg_loss,
            output_variance=avg_var,
            duration_sec=duration,
            accuracy=train_accuracy,
        )

    # ------------------------------------------------------------------
    def validate_epoch(self, loader: DataLoader) -> EpochMetrics:
        self.model.eval()
        preds: List[torch.Tensor] = []
        gts: List[torch.Tensor] = []
        running_loss = 0.0
        sample_count = 0
        start_time = time.time()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Val", leave=False):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                outputs = outputs.float()
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                sample_count += inputs.size(0)
                if self.is_classification:
                    probs = torch.sigmoid(outputs)
                    preds_bin = (probs >= 0.5).float()
                    correct += (preds_bin == targets).sum().item()
                    total += targets.numel()
                preds.append(outputs.detach().cpu())
                gts.append(targets.detach().cpu())
        duration = time.time() - start_time
        avg_loss = running_loss / max(sample_count, 1)
        accuracy = correct / total if self.is_classification and total > 0 else None
        if preds:
            pred_mat = torch.cat(preds, dim=0)
            gt_mat = torch.cat(gts, dim=0)
            pred_vec = pred_mat.reshape(-1)
            gt_vec = gt_mat.reshape(-1)
            if self.is_classification:
                prob_vec = torch.sigmoid(pred_vec)
                correlation = None
                abs_err = None
                variance = torch.var(prob_vec).item()
            else:
                correlation = self._pearson_corrcoef(pred_vec, gt_vec)
                abs_err = torch.mean(torch.abs(pred_vec - gt_vec)).item()
                variance = torch.var(pred_vec).item()
        else:
            correlation = None
            abs_err = None
            variance = None
        if variance is not None and variance < 1e-4:
            print("[Warning] Validation output variance below threshold. Mode collapse suspected.")
        return EpochMetrics(
            loss=avg_loss,
            correlation=correlation,
            output_variance=variance,
            abs_error=abs_err,
            duration_sec=duration,
            accuracy=accuracy,
        )

    # ------------------------------------------------------------------
    def _pearson_corrcoef(self, preds: torch.Tensor, gts: torch.Tensor) -> Optional[float]:
        if preds.numel() < 2:
            return None
        preds = preds - preds.mean()
        gts = gts - gts.mean()
        denom = torch.sqrt(torch.sum(preds ** 2) * torch.sum(gts ** 2))
        if denom == 0:
            return None
        return (torch.sum(preds * gts) / denom).item()

    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        latest: bool = True,
        extra_path: Path | None = None,
    ) -> None:
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict(),
            "epoch": epoch + 1,
            "best_val_loss": self.best_val_loss,
        }
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        if latest:
            torch.save(state, latest_path)
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_checkpoint.pth")
        if extra_path is not None:
            torch.save(state, extra_path)

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state and state["optimizer_state"] is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler and state.get("scheduler_state"):
            self.scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state and state["scaler_state"] is not None:
            if hasattr(self.scaler, "load_state_dict"):
                self.scaler.load_state_dict(state["scaler_state"])
        self.start_epoch = int(state.get("epoch", 0))
        self.best_val_loss = float(state.get("best_val_loss", math.inf))
        print(f"Resumed training from epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    def _log_epoch_summary(
        self, epoch: int, train_metrics: EpochMetrics, val_metrics: Optional[EpochMetrics]
    ) -> None:
        msg = [
            f"Epoch {epoch + 1}",
            f"train_loss={train_metrics.loss:.4f}",
        ]
        if train_metrics.output_variance is not None:
            msg.append(f"train_var={train_metrics.output_variance:.4f}")
        if train_metrics.accuracy is not None:
            msg.append(f"train_acc={train_metrics.accuracy:.4f}")
        if val_metrics is not None:
            msg.append(f"val_loss={val_metrics.loss:.4f}")
            if val_metrics.correlation is not None:
                msg.append(f"val_corr={val_metrics.correlation:.4f}")
            if val_metrics.output_variance is not None:
                msg.append(f"val_var={val_metrics.output_variance:.4f}")
            if val_metrics.abs_error is not None:
                msg.append(f"val_abs_err={val_metrics.abs_error:.4f}")
            if val_metrics.accuracy is not None:
                msg.append(f"val_acc={val_metrics.accuracy:.4f}")
        msg.append(f"epoch_time={train_metrics.duration_sec:.1f}s")
        print(" | ".join(msg))

    # ------------------------------------------------------------------
    def _save_metric_plots(self, current_epoch: int) -> None:
        epochs = range(1, len(self.history.train) + 1)
        train_losses = [m.loss for m in self.history.train]
        val_losses = [m.loss for m in self.history.val]
        val_epochs = range(1, len(val_losses) + 1)
        val_corr_points = [
            (idx + 1, m.correlation)
            for idx, m in enumerate(self.history.val)
            if m.correlation is not None
        ]
        val_var_points = [
            (idx + 1, m.output_variance)
            for idx, m in enumerate(self.history.val)
            if m.output_variance is not None
        ]

        fig, axes = plt.subplots(2, 1, figsize=(7, 8))

        axes[0].plot(list(epochs), train_losses, label="Train Loss", color="tab:blue")
        if val_losses:
            axes[0].plot(list(val_epochs), val_losses, label="Val Loss", color="tab:orange")
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[0].legend()

        axes[1].set_title("Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        if val_corr_points:
            axes[1].plot(
                [p[0] for p in val_corr_points],
                [p[1] for p in val_corr_points],
                label="Correlation",
                color="tab:green",
            )
        if val_var_points:
            axes[1].plot(
                [p[0] for p in val_var_points],
                [p[1] for p in val_var_points],
                label="Output Variance",
                color="tab:purple",
            )
        if not val_corr_points and not val_var_points:
            axes[1].text(0.5, 0.5, "No validation metrics yet", ha="center", va="center")
        else:
            axes[1].legend()

        plt.tight_layout()
        plot_path = self.checkpoint_dir / f"metrics_epoch_{current_epoch:03d}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved metric plot to {plot_path}")

    # ------------------------------------------------------------------
    def _correlation_penalty(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        preds_centered = preds_flat - preds_flat.mean()
        targets_centered = targets_flat - targets_flat.mean()
        pred_var = torch.sum(preds_centered**2)
        target_var = torch.sum(targets_centered**2)
        denom = torch.sqrt(pred_var * target_var) + self.corr_epsilon
        if not torch.isfinite(denom):
            return torch.zeros((), device=preds.device, dtype=preds.dtype)
        corr = torch.sum(preds_centered * targets_centered) / denom
        return 1.0 - corr


__all__ = ["Trainer", "TrainingHistory", "EpochMetrics"]
