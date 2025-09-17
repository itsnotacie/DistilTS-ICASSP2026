import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.metrics import metric

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class Exp_TimeMoe(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    @torch.no_grad()
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)       # [B, L, D]
            batch_y = batch_y.float().to(self.device)       # [B, L+pred, D]
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            if getattr(self.args, 'use_amp', False):
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, :]                   # [B, pred_len, D_all]
            batch_y2 = batch_y[:, -self.args.pred_len:, :].to(self.device)  # [B, pred_len, D_all]

            outputs = outputs.detach().cpu().numpy()
            batch_y2 = batch_y2.detach().cpu().numpy()

            if getattr(test_data, 'scale', False) and getattr(self.args, 'inverse', False):
                shape = batch_y2.shape
                if outputs.shape[-1] != batch_y2.shape[-1]:
                    outputs = np.tile(outputs, [1, 1, int(batch_y2.shape[-1] / outputs.shape[-1])])
                outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                batch_y2 = test_data.inverse_transform(batch_y2.reshape(shape[0] * shape[1], -1)).reshape(shape)

            outputs = outputs[:, :, f_dim:]
            batch_y2 = batch_y2[:, :, f_dim:]

            preds.append(outputs)
            trues.append(batch_y2)

        preds = np.concatenate(preds, axis=0)  # [N, pred_len, D']
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # 与原 metrics 期望的形状保持一致
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 结果保存与指标
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe, _ = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            f.write('\n\n')

        self.profile_model(test_loader)

        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'mspe': mspe}

    @torch.no_grad()
    def profile_model(self, test_loader):
        self.model.eval()

        if self.args.model == "Chronos":
            total_params = sum(p.numel() for p in self.model._pipe.model.parameters())
            print(f"Total parameters: {total_params/1e6:.2f}M")
            print(f"Total parameters: {total_params:.2f}")


        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.time()
        _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        infer_time = end_time - start_time
        gpu_mem = (torch.cuda.memory_allocated(self.device) / 1024 / 1024) if torch.cuda.is_available() else 0.0
        peak_mem = (torch.cuda.max_memory_allocated(self.device) / 1024 / 1024) if torch.cuda.is_available() else 0.0
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("=" * 80)
        print("TimeMoE Eval Profiling")
        print(f"{'Total Params':<25}: {total_params:,}")
        print(f"{'Inference Time (s)':<25}: {infer_time:.6f}")
        print(f"{'GPU Mem Footprint (MB)':<25}: {gpu_mem:.2f}")
        print(f"{'Peak Mem (MB)':<25}: {peak_mem:.2f}")
        print("=" * 80)
