from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import torch.nn.functional as F
import math

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')


class Exp_DistilTS(Exp_Basic):
    def __init__(self, args):
        super(Exp_DistilTS, self).__init__(args)
        
        if self.args.TSFModel == "TimeMoe":
            from models.TimeMoe import Model
        elif self.args.TSFModel == "Moirai":
            from models.Moirai import Model
        elif self.args.TSFModel == "Chronos":
            from models.Chronos import Model
        else:
            Model = None
            
        self.teacher = None
        self.kd_alpha = args.kd_alpha

        if self.kd_alpha > 0:
            self.teacher = Model(args).to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        
        if self.args.vt_loss:
            self.vt_aligner = VarTimeFactorAligner(
                T=args.seq_len, s_dim=args.d_model, t_dim=args.moe_dim, nonlin="gelu", loss_type="mse"
            ).to(self.device)


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def _horizon_weights(self, T: int, device):
        tau = float(self.args.horizon_weight_tau)
        if tau <= 0:
            return None
        t = torch.arange(T, device=device, dtype=torch.float32)
        w = torch.exp(tau * (t / max(T-1, 1)))
        # w = 1.0 + tau * (t / max(T-1, 1))
        return w / (w.mean() + 1e-8)

    def _weighted_mse(self, a, b, w_t=None):
        # a,b: [B,T,C]
        e2 = (a - b) ** 2
        if w_t is not None:
            e2 = e2 * w_t.view(1, -1, 1)
        return e2.mean()
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        # kd loss
                        kd_loss = 0.0
                        if self.teacher is not None and self.kd_alpha > 0.0:
                            with torch.no_grad():
                                t_outputs = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                t_outputs = t_outputs[:, -self.args.pred_len:, f_dim:]
                            # kd_loss = criterion(outputs, t_outputs.float())
                            kd_loss = self._weighted_mse(outputs, t_outputs.float())

                        sup_loss = criterion(outputs, batch_y)
                        # loss = (1.0 - self.kd_alpha) * sup_loss + self.kd_alpha * kd_loss
                        loss = sup_loss + self.kd_alpha * kd_loss
                        
                        # loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    kd_loss = 0.0
                    if self.teacher is not None and self.kd_alpha > 0.0:
                        with torch.no_grad():
                            t_outputs = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            t_outputs = t_outputs[:, -self.args.pred_len:, f_dim:]
                        # kd_loss = criterion(outputs, t_outputs.float())
                        kd_loss = self._weighted_mse(outputs, t_outputs.float())

                    sup_loss = criterion(outputs, batch_y)
                    # loss = (1.0 - self.kd_alpha) * sup_loss + self.kd_alpha * kd_loss
                    loss = sup_loss + self.kd_alpha * kd_loss
                    
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.vt_loss:
                    vt_loss = self.vt_aligner(self.teacher._last_hidden, self.model.hidden_dim, block_T=128)
                    loss += 0.3 * vt_loss
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, _ = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        self.profile_model(test_loader)
        
        # best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        # if os.path.exists(best_model_path):
        #     os.remove(best_model_path)
        #     print(f"Deleted model checkpoint at: {best_model_path}")

        return
    
    def profile_model(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()

            _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            torch.cuda.synchronize()
            end_time = time.time()

            inference_time = end_time - start_time
            gpu_mem = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print("=" * 80)
            print("Model Profiling Summary")
            print(f"{'Total Params':<25}: {total_params:,}")
            print(f"{'Inference Time (s)':<25}: {inference_time:.6f}")
            print(f"{'GPU Mem Footprint (MB)':<25}: {gpu_mem:.2f}")
            print(f"{'Peak Mem (MB)':<25}: {peak_mem:.2f}")
            print("=" * 80)


class VarTimeFactorAligner(nn.Module):
    def __init__(self, T=512, s_dim=512, t_dim=384, u=256, nonlin="gelu", loss_type="mse"):
        super().__init__()
        self.T = T
        self.s_dim = s_dim
        self.t_dim = t_dim
        self.u = u
        self.loss_type = loss_type

        self.proj_s = nn.Linear(s_dim, u, bias=False)
        self.time_embed = nn.Parameter(torch.randn(T, u) / math.sqrt(u))
        self.proj_out = nn.Linear(u, t_dim, bias=False)

        if nonlin == "gelu":
            self.act = nn.GELU()
        elif nonlin == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self,
                teacher_hidden_cpu: torch.Tensor,  # [B*D, T, t_dim] on CPU/float32
                student_hidden: torch.Tensor,      # [B, D, s_dim]   on GPU
                block_T: int = 128) -> torch.Tensor:

        device = student_hidden.device
        dtype  = student_hidden.dtype

        B, D, s_dim = student_hidden.shape
        N = B * D

        S_u = self.proj_s(student_hidden)   # [B, D, u]

        total_loss = 0.0
        blocks = 0

        for st in range(0, self.T, block_T):
            ed = min(st + block_T, self.T)

            E_blk = self.time_embed[st:ed, :].to(device=device, dtype=dtype)     # [block, u]
            Z = self.act(S_u[:, :, None, :] * E_blk[None, None, :, :])           # [B, D, block, u]
            pred_blk = self.proj_out(Z)                                          # [B, D, block, t_dim]

            tgt_blk = teacher_hidden_cpu.view(B, D, self.T, self.t_dim)[:, :, st:ed, :] \
                                   .to(device=device, dtype=dtype, non_blocking=True)  # [B, D, block, t_dim]

            if self.loss_type == "cos":
                pred_n = F.normalize(pred_blk, dim=-1, eps=1e-8)
                tgt_n  = F.normalize(tgt_blk,  dim=-1, eps=1e-8)
                loss_blk = 1.0 - (pred_n * tgt_n).sum(dim=-1).mean()
            else:
                loss_blk = F.mse_loss(pred_blk, tgt_blk)

            total_loss += loss_blk
            blocks += 1

            del E_blk, Z, pred_blk, tgt_blk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss / blocks