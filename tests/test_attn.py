import torch 
from trackastra.model.model_parts import RelativePositionalAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = RelativePositionalAttention(coord_dim=2, 
                                        embed_dim=64,
                                        n_head=1,
                                        mode='rope',
                                        attn_dist_mode='v0')
    
    model.to(device)
        
    B,N = 3,11
    q = torch.rand(B,N,64).to(device)
    x = torch.rand(B,N,3).to(device)
    
    pad_mask = torch.zeros((B, N), dtype=torch.bool).to(device)
    pad_mask[0,-2:] = True
    pad_mask[1,-3:] = True
    pad_mask[2,-4:] = True
    
    
    u1 = model(q[:1],q[:1],q[:1],coords=x[:1])  

    u = model(q,q,q,coords=x, padding_mask=pad_mask)

    print(torch.allclose(u[:1],u1, atol=1e-6))