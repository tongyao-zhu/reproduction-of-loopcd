"""Independent streaming graph: separate native cache objects, no slot wrapper."""

def oracle_stream(model,ids,state=None):
    import torch
    import copy
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    if state is None:state=dict(length=0,caches=[DynamicCache() for _ in range(11)])
    net=model.model;x=net.embed_tokens(ids);pos=torch.arange(state['length'],state['length']+ids.shape[1],device=ids.device)
    # The prelude's native layer zero supplies the actual past length.
    # Ask the native mask builder for an explicit mask. SDPA's automatic
    # mask elision selects a different GPU kernel and changes BF16 rounding.
    # This copy affects only mask construction, not model attention or weights.
    mask_config=copy.copy(net.config);mask_config._attn_implementation='eager'
    mask=create_causal_mask(config=mask_config,input_embeds=x,attention_mask=None,cache_position=pos,
        past_key_values=state['caches'][0],position_ids=pos[None])
    rope=net.rotary_emb(x,pos[None])
    def layers(indices,value,cache):
        for i in indices:
            value=net.layers[i](value,attention_mask=mask,position_ids=pos[None],past_key_value=cache,
                use_cache=True,cache_position=pos,position_embeddings=rope)
        return value
    x=layers(range(15),x,state['caches'][0]);states=[]
    for step in range(8):
        x=.875*x+.125*layers(range(15,19),x,state['caches'][1+step]);states.append(x)
    outputs=[]
    for i,hidden in enumerate((states[7],states[0])):
        outputs.append(model.lm_head(net.norm(layers(range(19,36),hidden,state['caches'][9+i]))).float())
    state['length']+=ids.shape[1]
    return outputs[0]+.3*(outputs[0]-outputs[1]),state
