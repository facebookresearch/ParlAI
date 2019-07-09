import os
import torch

FILE_TO_EDIT = '/checkpoint/edinan/wizard_release/model'

state_dict_map = {
    'persona_transformer': 'memory_transformer',
    'attentions.0': 'layers.0.attention',
    'attentions.1': 'layers.1.attention',
    'attentions.2': 'layers.2.attention',
    'attentions.3': 'layers.3.attention',
    'ffns.0': 'layers.0.ffn',
    'ffns.1': 'layers.1.ffn',
    'ffns.2': 'layers.2.ffn',
    'ffns.3': 'layers.3.ffn',
    'layer_norm1.0': 'layers.0.norm1',
    'layer_norm1.1': 'layers.1.norm1',
    'layer_norm1.2': 'layers.2.norm1',
    'layer_norm1.3': 'layers.3.norm1',
    'layer_norm2.0': 'layers.0.norm2',
    'layer_norm2.1': 'layers.1.norm2',
    'layer_norm2.2': 'layers.2.norm2',
    'layer_norm2.3': 'layers.3.norm2',
}

keys_to_delete = [
    'final_proj.bias',
    'final_proj.weight',
    'persona_encoder.proj.0.bias',
    'persona_encoder.proj.0.weight',
    'persona_encoder.proj.2.bias',
    'persona_encoder.proj.2.weight',
    'persona_encoder.proj.4.bias',
    'persona_encoder.proj.4.weight',
    'persona_encoder.embedding.weight',
]

with open(FILE_TO_EDIT, 'rb') as f:
    model = torch.load(f)

old_state_dict = model['model']
new_state_dict = {}
for k, v in old_state_dict.items():
    if k not in keys_to_delete:
        new_k = k
        if 'module.' in new_k:
            new_k = new_k[7:]
        for u, w in state_dict_map.items():
            if u in new_k:
                new_k = new_k.replace(u, w)
        new_state_dict[new_k] = v

model['model'] = new_state_dict

with open(FILE_TO_EDIT, 'wb') as g:
    torch.save(model, g)
