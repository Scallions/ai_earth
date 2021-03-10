from Informer2020.models.model import Informer
import torch
from constants import *


def build_model():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = DEVICE
    model = Informer(
        4, # args.enc_in,
        4, #args.dec_in, 
        1, #args.c_out, 
        12, #args.seq_len, 
        12, #args.label_len,
        26, #args.pred_len, 
        1, #args.factor,
        512, #args.d_model, 
        8, #args.n_heads, 
        3, #args.e_layers,
        2, #args.d_layers, 
        512, #args.d_ff,
        0.05, #args.dropout, 
        'prob', #args.attn,
        'timeF', #args.embed, timeF : month of year fixed: 1-12
        'm', #args.freq,
        'gelu', #args.activation,
        True, #args.output_attention,
        True, #args.distil,
        device
    )
    return model