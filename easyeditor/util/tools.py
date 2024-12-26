

def get_model_key(model_name):
    # by default, put all module on cuda:0
    model_name = model_name.lower()
    device_map = {}
    if 'gpt-j-6b' in model_name:
        key = ['transformer.wte', 'transformer.drop'] + ['transformer.h.'+str(i) for i in range(28)] + ['transformer.ln_f', 'lm_head']
    elif 'gpt2-xl' in model_name:
        key = ['transformer.wte', 'transformer.wpe'] + ['transformer.h.'+str(i) for i in range(48)] + ['transformer.ln_f', 'lm_head']
    elif 'opt-2.7b' in model_name:
        key = ['model.decoder.embed_tokens', 'model.decoder.embed_positions', 'lm_head'] + ['model.decoder.layers.'+str(i) for i in range(32)] + ['model.decoder.self_attn_layer_norm', 'model.decoder.fc1', 'model.decoder.fc2', 'model.decoder.final_layer_norm']
    elif 'opt-125m' in model_name:
        key = ['model.decoder.embed_tokens', 'model.decoder.embed_positions', 'lm_head'] + ['model.decoder.layers.'+str(i) for i in range(12)] + ['model.decoder.self_attn_layer_norm', 'model.decoder.fc1', 'model.decoder.fc2', 'model.decoder.final_layer_norm']
    elif 'chatglm2_6b' in model_name:
        key = ['transformer.embedding', 'transformer.rotary_pos_emb'] + ['transformer.encoder.layers.'+str(i) for i in range(28)] + ['transformer.encoder.final_layernorm', 'transformer.output_layer']
    elif 'llama-2-7b' in model_name:
        key = ['model.embed_tokens'] + ['model.layers.'+str(i) for i in range(32)] + ['model.norm', 'lm_head']
    elif 'vicuna-7b' in model_name:
        key = ['model.embed_tokens'] + ['model.layers.'+str(i) for i in range(32)] + ['model.norm', 'lm_head']
    elif 'vicuna-13b' in model_name:
        key = ['model.embed_tokens'] + ['model.layers.'+str(i) for i in range(40)] + ['model.norm', 'lm_head']
    else:
        return 'auto'
    for k in key:
        device_map[k] = 0
    return device_map

def get_device_map(device_map, hparams):
    # change the exist device_map, and set module by module name and split in hparams
    if device_map == 'auto':
        return device_map
    index = 0
    for (k, _) in device_map.items():
        if len(hparams.gpu_split) == 0:
            device_map[k] = hparams.device
        else:
            for j in range(len(hparams.gpu_split)):
                if hparams.gpu_split[j] in k and k[-2] == hparams.gpu_split[j][-2]:
                    index += 1
            device_map[k] = hparams.gpu_used_id[index]
    print(device_map)
    return device_map

def print_dict_info(d):
    for key in d.keys():
        if 'image' in key:
            print(key,': ',d[key].shape)
        else:
            print(key,': ',d[key])
            
def get_model_intervene_module_name(model_name):
    res = []
    if 'opt-2.7b' in model_name:
        for i in range(32):
            layer = 'model.decoder.layers[{}]'.format(str(i))
            res += [layer + '.self_attn', layer + '.fc2']
        res += ['model.decoder.fc2']
    elif 'opt-125m' in model_name:
        for i in range(12):
            layer = 'model.decoder.layers[{}]'.format(str(i))
            res += [layer + '.self_attn', layer + '.fc2']
        res += ['model.decoder.fc2']
    elif 'llama-2-7b' in model_name:
        for i in range(32):
            layer = 'model.layers[{}]'.format(str(i))
            res += [layer + '.self_attn', layer + '.mlp']
    elif 'vicuna-7b' in model_name:
        for i in range(32):
            layer = 'model.layers[{}]'.format(str(i))
            res += [layer + '.self_attn', layer + '.mlp']
    elif 'vicuna-13b' in model_name:
        for i in range(40):
            layer = 'model.layers[{}]'.format(str(i))
            res += [layer + '.self_attn', layer + '.mlp']
    else:
        print('Not define the layers to intervene!')
    res_module = [i.replace('[', '.') for i in res]
    res_module = [i.replace(']', '') for i in res_module]
    return res, res_module