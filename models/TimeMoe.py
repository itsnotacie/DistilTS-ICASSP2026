import logging
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.pred_len: int = configs.pred_len
        self.context_length: int = configs.seq_len

        self.register_buffer('_device_anchor', torch.empty(0))

        model_path: str = configs.pretrained_path

        inner = None
        inner = AutoModelForCausalLM.from_pretrained(
            model_path,
            # device_map="auto" if torch.cuda.is_available() else None,
            device_map=None,
            torch_dtype='auto',
            trust_remote_code=True,
        )

        self.gen_chunk = configs.gen_chunk
        
        logging.info(
            f"[TimeMoE-Adapter] dtype={getattr(inner, 'dtype', 'unknown')}; "
            f"attn={getattr(getattr(inner, 'config', None), '_attn_implementation', 'unknown')}"
        )
        self.inner = inner
        self.inner.eval()

    @property
    def device(self):
        return self._device_anchor.device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.inner.to(self.device)
        return self


    # @torch.no_grad()
    # def _generate_for_channels(self, x_ctx_2d: torch.Tensor) -> torch.Tensor:
    #     """
    #     x_ctx_2d: [N, ctx]  其中 N=B*D
    #     return:   [N, pred_len]
    #     """
    #     model = self.inner
    #     N = x_ctx_2d.size(0)
    #     dtype = getattr(model, 'dtype', torch.float32)

    #     if getattr(self, 'gen_chunk', 0) and 0 < self.gen_chunk < N:
    #         outs = []
    #         for s in range(0, N, self.gen_chunk):
    #             e = min(s + self.gen_chunk, N)
    #             chunk = x_ctx_2d[s:e].to(self.device).to(dtype)
    #             out = model.generate(inputs=chunk, max_new_tokens=self.pred_len)
    #             outs.append(out[:, -self.pred_len:])
    #             del chunk, out
    #             if torch.cuda.is_available():
    #                 torch.cuda.empty_cache()
    #         preds = torch.cat(outs, dim=0)
    #     else:
    #         out = model.generate(
    #             inputs=x_ctx_2d.to(self.device).to(dtype),
    #             max_new_tokens=self.pred_len,
    #         )
    #         preds = out[:, -self.pred_len:]

    #     return preds
    
    @torch.no_grad()
    def _get_model_dtype(self):
        # 兼容性：优先 inner.dtype，否则取首个参数的 dtype
        if self.inner.dtype is not None:
            return self.inner.dtype
        for p in self.inner.parameters():
            return p.dtype
        return torch.float32

    @torch.no_grad()
    def _generate_for_channels(self, x_ctx_2d: torch.Tensor, return_hidden: bool = True):
        model = self.inner
        N = x_ctx_2d.size(0)
        dtype = self._get_model_dtype()

        def gen_one(chunk_2d: torch.Tensor):
            out = model.generate(
                inputs=chunk_2d.to(self.device).to(dtype),  # 2D: [n, ctx]
                max_new_tokens=self.pred_len,
                return_dict_in_generate=True,        
                output_hidden_states=return_hidden,         
            )
            preds = out.sequences[:, -self.pred_len:]       # [n, pred_len]

            if not return_hidden:
                return preds, None

            prefill_hidden = out.hidden_states[0][-1]
            prefill_hidden = prefill_hidden.detach().to(torch.float32).cpu()
            return preds, prefill_hidden

        if getattr(self, 'gen_chunk', 0) and 0 < self.gen_chunk < N:
            preds_list = []
            hids_list = [] if return_hidden else None
            for s in range(0, N, self.gen_chunk):
                e = min(s + self.gen_chunk, N)
                p, h = gen_one(x_ctx_2d[s:e])
                preds_list.append(p)
                if return_hidden:
                    hids_list.append(h)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            preds = torch.cat(preds_list, dim=0)
            hidden = torch.cat(hids_list, dim=0) if return_hidden else None
        else:
            preds, hidden = gen_one(x_ctx_2d)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (preds, hidden) if return_hidden else preds

    @torch.no_grad()
    def forward(self,
                x_enc: torch.Tensor,
                x_mark_enc: Optional[torch.Tensor] = None,
                x_dec: Optional[torch.Tensor] = None,
                x_mark_dec: Optional[torch.Tensor] = None,
                mask=None) -> torch.Tensor:

        B, L, D = x_enc.shape
        ctx = min(int(self.context_length), int(L))
        
        x_ctx = x_enc[:, -ctx:, :].permute(0, 2, 1).contiguous().view(B * D, ctx)

        # preds_flat = self._generate_for_channels(x_ctx)

        preds_flat, hidden = self._generate_for_channels(x_ctx, return_hidden=True)
        # 缓存起来，外部如果需要蒸馏/对齐，从这里拿
        self._last_hidden = hidden
        preds = preds_flat.view(B, D, self.pred_len).permute(0, 2, 1).contiguous()
        return preds 
