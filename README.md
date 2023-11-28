[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Attention + Convolution transformer
This is an experimental architecture leveraging convolution blocks with attention blocks to model both the short and long range dynamics of the input tokens. The flow is the following: x -> convolution block -> attn -> FFN

# Install
``


## Usage
```python
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

```


# License
MIT



