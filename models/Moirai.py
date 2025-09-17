import logging
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiModule, MoiraiForecast


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.pred_len: int = int(configs.pred_len)
        self.context_length: int = int(configs.seq_len)
        self._patch_size_cfg = configs.patch_size
        self.gen_chunk: int = int(configs.gen_chunk)
        self.freq: str = configs.freq
        self._ext_batch_size: int = configs.batch_size

        self.register_buffer("_device_anchor", torch.empty(0))
        model_path: str = configs.pretrained_path

        self.patch_size = configs.patch_size

        module = MoiraiModule.from_pretrained(model_path)
        module.eval()
        self.inner: MoiraiModule = module

        self._forecast = MoiraiForecast(
            module=self.inner,
            prediction_length=self.pred_len,
            context_length=self.context_length,
            patch_size=self.patch_size,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = None

        logging.info(
            f"[Moirai-Adapter] loaded from {model_path}; "
            f"ctx={self.context_length}, pred={self.pred_len}, patch_size={self.patch_size} "
            f"(from cfg={self._patch_size_cfg})"
        )

    @property
    def device(self):
        return self._device_anchor.device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.inner.to(self.device)
        self._predictor = self._forecast.create_predictor(
            batch_size=self._ext_batch_size,
            device="cuda" if (torch.cuda.is_available() and self.device.type == "cuda") else "cpu",
        )
        return self

    def _build_listdataset(self, series_2d: np.ndarray) -> ListDataset:
        """series_2d: [N, ctx] → GluonTS ListDataset"""
        start = pd.Timestamp("2000-01-01 00:00:00")
        data = [{"start": start, "target": row.astype(np.float32)} for row in series_2d]
        return ListDataset(data, freq=self.freq)

    def _predict_ndarray(self, series_2d: np.ndarray) -> np.ndarray:
        ds = self._build_listdataset(series_2d)
        preds: List[np.ndarray] = []
        for fcst in self._predictor.predict(ds):
            preds.append(fcst.mean.astype(np.float32))  # (pred_len,)
        return np.stack(preds, axis=0)

    @torch.no_grad()
    def forward(
        self,
        x_enc: torch.Tensor,                  # [B, L, D]
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        mask=None,
    ) -> torch.Tensor:
        assert x_enc.ndim == 3, f"x_enc must be [B, L, D], got {x_enc.shape}"
        B, L, D = x_enc.shape
        ctx = min(self.context_length, L)

        # [B, L, D] → [N=B*D, ctx] (CPU numpy for GluonTS)
        x_ctx = (
            x_enc[:, -ctx:, :]
            .permute(0, 2, 1)      # [B, D, ctx]
            .contiguous()
            .view(B * D, ctx)      # [N, ctx]
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float32)
        )

        if self.gen_chunk and 0 < self.gen_chunk < (B * D):
            outs = []
            for s in range(0, B * D, self.gen_chunk):
                e = min(s + self.gen_chunk, B * D)
                outs.append(self._predict_ndarray(x_ctx[s:e]))  # [chunk, pred_len]
            preds_2d = np.concatenate(outs, axis=0)
        else:
            preds_2d = self._predict_ndarray(x_ctx)             # [N, pred_len]

        preds = (
            torch.from_numpy(preds_2d)
            .to(self.device)
            .view(B, D, self.pred_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        
        return preds