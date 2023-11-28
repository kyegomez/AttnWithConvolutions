import torch
from attnconv.main import ATCTransformer

model = ATCTransformer(
    dim=512,
    depth=6,
    num_tokens=20000,
    dim_head=64,
    heads=8,
    ff_mult=4,
)

x = torch.randint(0, 20000, (1, 512))
logits = model(x)  # (1, 1024, 20000)
print(logits)
