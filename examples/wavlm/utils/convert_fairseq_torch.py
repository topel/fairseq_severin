import os
import torch
import fairseq
import torchaudio
from fairseq import checkpoint_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from torch.nn import Module
from torchaudio.models.wav2vec2 import Wav2Vec2Model, wav2vec2_model
import sys

def fairseq_pt_torch_ckpt(fairseq_pt):
    
    ckpt = {} #Create the new dictionary
    print(fairseq_pt)
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_pt]) #Instantiate the fairseq model
    model = models[0]
    model.eval()
    
    ckpt['model_name'] = model.__class__.__name__ #Add the model name to the new dictionary
    cfg = _parse_config(model) #Extract the configuration from the fairseq model and convert it to torchaudio format
    cfg['aux_num_out'] = None #No finetuning involved so has to be set to None
    ckpt['config']=cfg #Add the configuration file
    
    print("Configuration saved")
    
    model.__class__.__name__ = "Wav2Vec2Model" #Rename the model name to allow conversion to torchaudio
    model = import_fairseq_model(model).eval() #Convert the fairseq model to torchaudio
    
    dict = model.state_dict() #Get the torchaudio dictionary
    
    ckpt['state_dict']=dict #Add the torchaudio dictionary to the ckpt
    
    print("State dict saved")
    
    torch.save(ckpt, (os.path.splitext(fairseq_pt)[0])+'.ckpt') #Save the new dictionary  
    
    print((os.path.splitext(fairseq_pt)[0])+'.ckpt created !')
    
    
def _parse_config(w2v_model):
    encoder = w2v_model.encoder
    conv_layers = w2v_model.feature_extractor.conv_layers

    extractor_mode = "layer_norm"
    if "GroupNorm" in conv_layers[0][2].__class__.__name__:
        extractor_mode = "group_norm"
    else:
        extractor_mode = "layer_norm"

    conv_layer_config = [(l[0].out_channels, l[0].kernel_size[0], l[0].stride[0]) for l in conv_layers]

    if all(l[0].bias is None for l in conv_layers):
        conv_bias = False
    elif all(l[0].bias is not None for l in conv_layers):
        conv_bias = True
    else:
        raise ValueError("Either all the convolutions layers have bias term or none of them should.")

    config = {
        "extractor_mode": extractor_mode,
        "extractor_conv_layer_config": conv_layer_config,
        "extractor_conv_bias": conv_bias,
        "encoder_embed_dim": w2v_model.post_extract_proj.out_features,
        "encoder_projection_dropout": w2v_model.dropout_input.p,
        "encoder_pos_conv_kernel": encoder.pos_conv[0].kernel_size[0],
        "encoder_pos_conv_groups": encoder.pos_conv[0].groups,
        "encoder_num_layers": len(encoder.layers),
        "encoder_num_heads": encoder.layers[0].self_attn.num_heads,
        "encoder_attention_dropout": encoder.layers[0].self_attn.dropout_module.p,
        "encoder_ff_interm_features": encoder.layers[0].fc1.out_features,
        "encoder_ff_interm_dropout": encoder.layers[0].dropout2.p,
        "encoder_dropout": encoder.layers[0].dropout3.p,
        "encoder_layer_norm_first": encoder.layer_norm_first,
        "encoder_layer_drop": encoder.layerdrop,
    }
    return config


if __name__ == "__main__":
    
    fairseq_pt_torch_ckpt(sys.argv[1])
