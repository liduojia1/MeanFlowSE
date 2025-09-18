import time
from math import ceil
import warnings
import numpy as np
import torch
from torch.autograd.functional import jvp as autograd_jvp
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
import torch.distributed as dist  
from flowmse import sampling
from flowmse.odes import ODERegistry
from flowmse.backbones import BackboneRegistry
from flowmse.util.inference import evaluate_model
from flowmse.util.other import pad_spec
import matplotlib.pyplot as plt
import random


class VFModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="t_delta in the paper")
        parser.add_argument("--T_rev", type=float, default=1.0, help="Starting point t_N in the paper")

        parser.add_argument("--num_eval_files", type=int, default=10,
                            help="Number of files for speech enhancement evaluation during training. Pass 0 to disable heavy eval (we will still log -inf placeholders to keep checkpoints working).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default=0.5, help="magnitude transformation in the loss term")

        parser.add_argument("--use_mfse", action="store_true", help="Enable MeanFlow-SE training (average velocity).")
        parser.add_argument("--mf_weight_final", type=float, default=0.25, help="Final weight of MFSE branch (average field).")
        parser.add_argument("--mf_warmup_frac", type=float, default=0.5, help="Fraction of max epochs to warm up MF weight from 0 to mf_weight_final.")
        parser.add_argument("--mf_delta_gamma_start", type=float, default=8.0, help="Initial exponent for sampling r near t: U**gamma (larger -> closer to t).")
        parser.add_argument("--mf_delta_gamma_end", type=float, default=1.0, help="Final exponent for sampling r near t.")
        parser.add_argument("--mf_delta_warmup_frac", type=float, default=0.7, help="Fraction of max epochs to anneal delta gamma.")
        parser.add_argument("--mf_r_equals_t_prob", type=float, default=0.1, help="Probability to force r=t samples in MFSE batch (stabilizes degeneration).")
        parser.add_argument("--mf_jvp_clip", type=float, default=5.0, help="Per-sample L2 clipping for JVP magnitude.")
        parser.add_argument("--mf_jvp_eps", type=float, default=1e-3, help="Relative step for finite-difference fallback of JVP.")
        parser.add_argument("--use_mf_sampler", action="store_true", help="Use MF (euler_mf) sampler in validation/test (enables 1-NFE).")

        parser.add_argument("--mf_jvp_impl", type=str, choices=("auto", "fd", "autograd"), default="auto",
                            help="JVP implementation: 'fd' (finite difference), 'autograd' (torch.autograd.functional.jvp), or 'auto' (try autograd then fallback to fd).")
        parser.add_argument("--mf_jvp_chunk", type=int, default=0,
                            help="Chunk size along batch dimension for FD-JVP (0=disable chunking, 1=per-sample).")
        parser.add_argument("--mf_skip_weight_thresh", type=float, default=0.0,
                            help="Skip MF branch entirely when current w_mf < this threshold to save memory.")

        parser.add_argument("--mf_first_order_coef", type=float, default=0.5,
                            help="Coefficient c in MF first-order correction: u_tgt = v_t - c * (t-r) * JVP. (0.5 = trapezoid-like; 1.0 = simple Taylor).")
        parser.add_argument("--val_metrics_every_n_epochs", type=int, default=1,
                            help="Run expensive evaluate_model every N epochs. <=0 means never (but placeholders will still be logged).")
        parser.add_argument("--val_metrics_num_files", type=int, default=None,
                            help="Override number of files used in evaluate_model during validation. If None, fall back to num_eval_files.")
        # ------------------------------------------------------------------
        return parser

    def __init__(
        self, backbone, ode, lr=1e-4, ema_decay=0.999, t_eps=0.03, T_rev=1.0, loss_abs_exponent=0.5,
        num_eval_files=10, loss_type='mse', data_module_cls=None,
        use_mfse=False, mf_weight_final=0.25, mf_warmup_frac=0.5,
        mf_delta_gamma_start=8.0, mf_delta_gamma_end=1.0, mf_delta_warmup_frac=0.7,
        mf_r_equals_t_prob=0.1, mf_jvp_clip=5.0, mf_jvp_eps=1e-3,
        use_mf_sampler=False,
        mf_jvp_impl="auto", mf_jvp_chunk=0, mf_skip_weight_thresh=0.0,
        mf_first_order_coef=0.5, val_metrics_every_n_epochs=20, val_metrics_num_files=None,
        **kwargs
    ):
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**{**kwargs, "scale_by_sigma": False})

        ode_cls = ODERegistry.get_by_name(ode)
        self.ode = ode_cls(**kwargs)

        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent

        # MFSE switches and schedules
        self.use_mfse = use_mfse
        self.mf_weight_final = mf_weight_final
        self.mf_warmup_frac = mf_warmup_frac
        self.mf_delta_gamma_start = mf_delta_gamma_start
        self.mf_delta_gamma_end = mf_delta_gamma_end
        self.mf_delta_warmup_frac = mf_delta_warmup_frac
        self.mf_r_equals_t_prob = mf_r_equals_t_prob
        self.mf_jvp_clip = mf_jvp_clip
        self.mf_jvp_eps = mf_jvp_eps
        self.use_mf_sampler = use_mf_sampler

        self.mf_jvp_impl = mf_jvp_impl
        self.mf_jvp_chunk = int(mf_jvp_chunk) if mf_jvp_chunk is not None else 0
        self.mf_skip_weight_thresh = float(mf_skip_weight_thresh)

        self.mf_first_order_coef = float(mf_first_order_coef)
        self.val_metrics_every_n_epochs = int(val_metrics_every_n_epochs) if val_metrics_every_n_epochs is not None else 0
        self.val_metrics_num_files = val_metrics_num_files if (val_metrics_num_files is None or val_metrics_num_files > 0) else None

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=torch.cuda.is_available())

    # ---------- Lightning boilerplate ----------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                self.ema.store(self.parameters())
                self.ema.copy_to(self.parameters())
            else:
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)
    # ------------------------------------------

    def _mse_loss(self, x, x_hat):
        err = x - x_hat
        losses = torch.square(err.abs())
        return torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

    def _loss(self, pred, target):
        if self.loss_type == 'mse':
            err = pred - target
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = pred - target
            losses = err.abs()
        else:
            raise ValueError(f"Unknown loss_type {self.loss_type}")
        return torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
    # ------------------------------------------

    def _progress(self):
        if not hasattr(self, "trainer") or (self.trainer is None) or (self.trainer.max_epochs is None):
            return 1.0
        maxe = max(1, int(self.trainer.max_epochs))
        return float(self.current_epoch) / float(maxe)

    def _mf_weight_now(self):
        p = self._progress()
        if p <= self.mf_warmup_frac:
            return self.mf_weight_final * (p / max(1e-9, self.mf_warmup_frac))
        else:
            return self.mf_weight_final

    def _delta_gamma_now(self):
        p = self._progress()
        frac = min(1.0, p / max(1e-9, self.mf_delta_warmup_frac))
        return self.mf_delta_gamma_start + frac * (self.mf_delta_gamma_end - self.mf_delta_gamma_start)

    def _sample_r_given_t(self, t):
        gamma = self._delta_gamma_now()
        u = torch.rand_like(t) ** gamma
        r = t - u * (t - self.t_eps)
        mask = (torch.rand_like(t) < self.mf_r_equals_t_prob)
        r = torch.where(mask, t, r)
        return r
    # ------------------------------------------

    def _fd_jvp_chunked(self, xt, t, y_const, r_const, v_dir, t_dir):
        """
        有限差分 JVP，按 batch 维分块以节省显存。
        返回：与 forward(x,t,y,r) 形状一致的张量（已 clip，已 detach）
        """
        B = xt.shape[0]
        out = torch.zeros_like(v_dir)
        mean_scale = (xt.abs().mean(dim=[1, 2, 3], keepdim=True) +
                      v_dir.abs().mean(dim=[1, 2, 3], keepdim=True)).clamp_min(1e-6)
        eps = self.mf_jvp_eps * mean_scale            # [B,1,1,1]
        eps_t = eps.view(-1)                          # [B]
        chunk = max(1, int(self.mf_jvp_chunk)) if self.mf_jvp_chunk else B

        from torch.cuda.amp import autocast
        with torch.no_grad(), autocast(enabled=False):
            for s in range(0, B, chunk):
                e = min(B, s + chunk)
                xt_s = xt[s:e]; t_s = t[s:e]; y_s = y_const[s:e]
                r_s = r_const[s:e] if r_const is not None else None
                vdir_s = v_dir[s:e]; tdir_s = t_dir[s:e]
                eps_s = eps[s:e]; eps_t_s = eps_t[s:e]

                u_pos = self.forward(xt_s + eps_s * vdir_s, t_s + eps_t_s * tdir_s, y_s, r_s)
                u_neg = self.forward(xt_s - eps_s * vdir_s, t_s - eps_t_s * tdir_s, y_s, r_s)
                out[s:e] = (u_pos - u_neg) / (2.0 * eps_s)

        flat = out.abs().reshape(B, -1)
        nrm = torch.linalg.norm(flat, ord=2, dim=1) + 1e-12
        scale = torch.clamp(self.mf_jvp_clip / nrm, max=1.0)
        out = out * scale[:, None, None, None]
        return out.detach()

    def _safe_jvp(self, xt, t, y, r, v_dir, t_dir):
        """
        计算 JVP: (v,1)·(∂_x u, ∂_t u) = v ∂_x u + ∂_t u
        - 'autograd'：torch.autograd.functional.jvp（更快但可能占显存）
        - 'fd'：中心差分（两次前向）；支持按 batch 分块
        - 'auto'：尝试 autograd，失败则回退 fd
        * 所有实现均返回已裁剪、detach 的张量（不反传二阶）
        """
        impl = getattr(self, "mf_jvp_impl", "auto")

        xt = xt.detach(); t = t.detach()
        y_const = y.detach()
        r_const = r.detach() if r is not None else None

        if impl == "autograd":
            xt_req = xt.requires_grad_(True); t_req = t.requires_grad_(True)
            def u_fn(xx, tt): return self.forward(xx, tt, y_const, r_const)
            _, jvp_val = autograd_jvp(u_fn, (xt_req, t_req), (v_dir.detach(), t_dir.detach()),
                                      create_graph=False, strict=False)
            B = jvp_val.shape[0]
            flat = jvp_val.abs().reshape(B, -1)
            nrm = torch.linalg.norm(flat, ord=2, dim=1) + 1e-12
            scale = torch.clamp(self.mf_jvp_clip / nrm, max=1.0)
            return (jvp_val * scale[:, None, None, None]).detach()

        if impl == "fd":
            return self._fd_jvp_chunked(xt, t, y_const, r_const, v_dir, t_dir)

        # impl == "auto"
        try:
            xt_req = xt.requires_grad_(True); t_req = t.requires_grad_(True)
            def u_fn(xx, tt): return self.forward(xx, tt, y_const, r_const)
            _, jvp_val = autograd_jvp(u_fn, (xt_req, t_req), (v_dir.detach(), t_dir.detach()),
                                      create_graph=False, strict=False)
            B = jvp_val.shape[0]
            flat = jvp_val.abs().reshape(B, -1)
            nrm = torch.linalg.norm(flat, ord=2, dim=1) + 1e-12
            scale = torch.clamp(self.mf_jvp_clip / nrm, max=1.0)
            return (jvp_val * scale[:, None, None, None]).detach()
        except Exception:
            return self._fd_jvp_chunked(xt, t, y_const, r_const, v_dir, t_dir)

    def _step(self, batch, batch_idx):
        x1, y = batch
        rdm = (1 - torch.rand(x1.shape[0], device=x1.device)) * (self.T_rev - self.t_eps) + self.t_eps
        t = torch.min(rdm, torch.tensor(self.T_rev, device=x1.device))

        mean, std = self.ode.marginal_prob(x1, t, y)
        z = torch.randn_like(x1)
        sigmas = std[:, None, None, None]
        xt = mean + sigmas * z

        der_std = self.ode.der_std(t)                     
        der_mean = self.ode.der_mean(x1, t, y)                  
        condVF = der_std[:, None, None, None] * z + der_mean    

        v_pred = self(xt, t, y, r=t)
        loss_cfm = self._loss(v_pred, condVF)

        if not self.use_mfse:
            self.log('train_loss_cfm', loss_cfm, on_step=True, on_epoch=True)
            self.log('train_loss_mf', torch.tensor(0.0, device=loss_cfm.device), on_step=True, on_epoch=True)
            self.log('train_w_mf', torch.tensor(0.0, device=loss_cfm.device), on_step=True, on_epoch=True)
            return loss_cfm

        w_mf = self._mf_weight_now()
        w_cfm = 1.0 - w_mf

        if w_mf < self.mf_skip_weight_thresh:
            loss = loss_cfm
            self.log('train_loss_cfm', loss_cfm, on_step=True, on_epoch=True)
            self.log('train_loss_mf', torch.tensor(0.0, device=loss_cfm.device), on_step=True, on_epoch=True)
            self.log('train_w_mf', torch.tensor(w_mf, device=loss_cfm.device), on_step=True, on_epoch=True)
            self.log('train_mf_skipped', torch.tensor(1.0, device=loss_cfm.device), on_step=True, on_epoch=True)
            return loss

        r = self._sample_r_given_t(t)
        jvp_val = self._safe_jvp(xt, t, y, r, v_dir=condVF, t_dir=torch.ones_like(t))
        delta = (t - r)[:, None, None, None]
        u_tgt = (condVF - self.mf_first_order_coef * delta * jvp_val).detach()
        u_pred = self(xt, t, y, r)
        loss_mf = self._loss(u_pred, u_tgt)

        loss = w_cfm * loss_cfm + w_mf * loss_mf

        self.log('train_loss_cfm', loss_cfm, on_step=True, on_epoch=True)
        self.log('train_loss_mf', loss_mf, on_step=True, on_epoch=True)
        self.log('train_w_mf', torch.tensor(w_mf, device=loss.device), on_step=True, on_epoch=True)
        self.log('train_mf_skipped', torch.tensor(0.0, device=loss.device), on_step=True, on_epoch=True)
        return loss

    def _should_run_val_metrics(self):
        if (self.num_eval_files is None) or (int(self.num_eval_files) <= 0):
            return False
        if (self.val_metrics_every_n_epochs is None) or (int(self.val_metrics_every_n_epochs) <= 0):
            return False
        return (int(self.current_epoch) % int(self.val_metrics_every_n_epochs) == 0)

    def _val_num_files_eff(self):
        if (self.val_metrics_num_files is not None) and (int(self.val_metrics_num_files) > 0):
            return int(self.val_metrics_num_files)
        return int(self.num_eval_files) if (self.num_eval_files is not None) else 0
    # -------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            run_eval = self._should_run_val_metrics()
            nfiles = self._val_num_files_eff()
            device = loss.device

            if run_eval and (nfiles > 0):
                if self.trainer.is_global_zero:
                    pesq, si_sdr, estoi = evaluate_model(self, nfiles)
                    m = torch.tensor([float(pesq), float(si_sdr), float(estoi)],
                                     device=device, dtype=torch.float32)
                else:
                    m = torch.zeros(3, device=device, dtype=torch.float32)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(m, src=0)
            else:
                m = torch.tensor([float('-inf'), float('-inf'), float('-inf')],
                                 device=device, dtype=torch.float32)

            self.log('pesq',  m[0].item(), on_step=False, on_epoch=True, sync_dist=False)
            self.log('si_sdr', m[1].item(), on_step=False, on_epoch=True, sync_dist=False)
            self.log('estoi', m[2].item(), on_step=False, on_epoch=True, sync_dist=False)

        return loss

    def forward(self, x, t, y, r=None):
        """
        统一前向：
        - 若 r is None -> Δ = 0 -> 与 FlowSE 瞬时场路径等价
        - 若 r 给定     -> Δ = t - r 作为第二时间条件注入 Backbone
        """
        dnn_input = torch.cat([x, y], dim=1)
        if r is None:
            d = torch.zeros_like(t)
        else:
            d = (t - r).clamp_min(0.0)
        score = -self.dnn(dnn_input, t, d)
        return score

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
