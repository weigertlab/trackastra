import torch 
from trackastra.model.model_parts import RelativePositionalAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = RelativePositionalAttention(coord_dim=2, 
                                        embed_dim=64,
                                        n_head=2,
                                        mode='rope',
                                        attn_dist_mode='v2')
    
    model.eval()
    model.to(device)
        
    B,N = 3,11
    q = torch.rand(B,N,64).to(device)
    x = torch.rand(B,N,3).to(device)
    
    pad_mask = torch.zeros((B, N), dtype=torch.bool).to(device)
    pad_mask[0,-2:] = True
    pad_mask[1,-3:] = True
    pad_mask[2,-4:] = True
    
    
    
    u = model(q,q,q,coords=x, padding_mask=pad_mask)
    mask = model.attn_mask

    u1 = model(q[:1],q[:1],q[:1],coords=x[:1],padding_mask=pad_mask[:1])  
    mask1 = model.attn_mask 

    err = torch.abs(u[:1] - u1).mean() 
    print(f'Error: {err:.4f}')
    print('close: ', torch.allclose(u[:1],u1, rtol=1e-3, atol=1e-6))
    
    