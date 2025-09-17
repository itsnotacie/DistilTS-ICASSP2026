import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_TimeMoe import Exp_TimeMoe




from utils.print_args import print_args
import random
import numpy as np
import sys
sys.dont_write_bytecode = True

os.environ['CUDA_VISIBLE_DEVICE']='0,1'

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='DistilTS')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--visualize_attn', action='store_true', help='Whether to visualize attention')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--channel', type=str, default='CI', help='CI or CD')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    # distilling
    parser.add_argument('--pretrained_path', type=str, default='../TimeMoE-50M', help='TFM weight')
    parser.add_argument('--gen_chunk', type=int, default=32, help='gen_chunk')
    parser.add_argument('--moe_dim', type=int, default=384, help='384 768')


    parser.add_argument('--kd_alpha', type=float, default=0.5, help='kd_alpha')

    # warm-up + huber loss
    parser.add_argument('--kd_min_alpha', type=float, default=0.1, help='KD warmup init')
    parser.add_argument('--kd_warmup_epochs', type=int, default=5, help='KD epochs')
    parser.add_argument('--kd_huber_delta', type=float, default=1.0, help='Huber loss delta')

    parser.add_argument('--horizon_weight_tau', type=float, default=1.0, help='>0')

    # frency
    parser.add_argument('--kd_freq_beta', type=float, default=1.0, help='main')
    parser.add_argument('--kd_freq_diff_gamma', type=float, default=0.1, help='diff')
    parser.add_argument('--kd_freq_eps', type=float, default=1e-3, help='s')

    # MORIAI
    parser.add_argument("--patch_size", type=int, default=64, help='patch size: "auto" or (8/16/32/64/128)')
    
    parser.add_argument('--TSFModel', type=str, default='TimeMoe', help='TimeMoe Moirai')

    parser.add_argument('--trend_kd_mode', type=str, default='stride', help='stride pool')
    
    parser.add_argument('--vt_loss', type=int, default=1, help='vt_loss')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'Exp_TimeMoe':
        Exp = Exp_TimeMoe
    elif args.task_name == 'Exp_DistilTS':
        from exp.exp_DistilTS import Exp_DistilTS
        Exp = Exp_DistilTS
    elif args.task_name == 'exp_DistilTS_TKD':
        from exp.exp_DistilTS_TKD import Exp_DistilTS
        Exp = Exp_DistilTS
    elif args.task_name == 'exp_DistilTS_FDKD':
        from exp.exp_DistilTS_FDKD import Exp_DistilTS
        Exp = Exp_DistilTS
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}_{}'.format(args.task_name, args.model_id, args.model, args.learning_rate, args.kd_alpha, args.TSFModel)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_{}'.format(args.task_name, args.model_id, args.model, args.learning_rate, args.kd_alpha, args.TSFModel)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()