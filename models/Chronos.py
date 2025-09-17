# -*- coding: utf-8 -*-
import logging
from typing import Optional

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.pred_len: int = int(configs.pred_len)
        self.context_length: int = int(configs.seq_len)
        self.gen_chunk: int = int(configs.gen_chunk)
        self.ext_bsz: int = int(configs.batch_size)
        self.use_bf16: bool = True
        self.model_id: str = str(configs.pretrained_path)

        self.register_buffer("_device_anchor", torch.empty(0))
        self._pipe = None

        logging.info(
            f"[Chronos-Adapter] prepared model_id={self.model_id} "
            f"ctx={self.context_length}, pred={self.pred_len}, "
            f"gen_chunk={self.gen_chunk}, ext_bsz={self.ext_bsz}"
        )

    @property
    def device(self):
        return self._device_anchor.device

    def _build_pipeline(self, device_str: str):
        try:
            from chronos import BaseChronosPipeline
        except Exception as e:
            raise RuntimeError(
                "chronos-forecasting please: pip install chronos-forecasting"
            ) from e

        dtype = torch.bfloat16 if (device_str == "cuda" and self.use_bf16) else torch.float32

        pipe = BaseChronosPipeline.from_pretrained(
            self.model_id,
            device_map=device_str, 
            torch_dtype=dtype,
        )
        return pipe

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dev = "cuda" if (torch.cuda.is_available() and self.device.type == "cuda") else "cpu"
        self._pipe = self._build_pipeline(dev)
        logging.info(f"[Chronos-Adapter] pipeline on {dev}, dtype={self._pipe.dtypes}")
        return self

    @torch.no_grad()
    def _predict_block(self, x_ctx_2d: torch.Tensor) -> torch.Tensor:

        quantiles, mean = self._pipe.predict_quantiles(
            context=x_ctx_2d,
            prediction_length=self.pred_len,
            quantile_levels=[0.5],
        )
        return mean.to(self.device)

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

        # [B, L, D] → [B, D, ctx] → [N, ctx]
        x_ctx_2d = (
            x_enc[:, -ctx:, :]          # [B, ctx, D]
            .permute(0, 2, 1)           # [B, D, ctx]
            .contiguous()
            .view(B * D, ctx)           # [N, ctx]
        )

        if self.gen_chunk and 0 < self.gen_chunk < (B * D):
            outs = []
            for s in range(0, B * D, self.gen_chunk):
                e = min(s + self.gen_chunk, B * D)
                for ss in range(s, e, self.ext_bsz):
                    ee = min(ss + self.ext_bsz, e)
                    outs.append(self._predict_block(x_ctx_2d[ss:ee]))
            preds_2d = torch.vstack(outs)  # [N, pred_len]
        else:
            outs = []
            N = B * D
            for s in range(0, N, self.ext_bsz):
                e = min(s + self.ext_bsz, N)
                outs.append(self._predict_block(x_ctx_2d[s:e]))
            preds_2d = torch.vstack(outs)

        # [N, pred_len] → [B, pred_len, D]
        preds = (
            preds_2d
            .view(B, D, self.pred_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        return preds
