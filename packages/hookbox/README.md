# hookbox

Utility library for capturing and transforming activations in PyTorch models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- Simple API for capture and transform
- Context-manager cleanup
- Efficient hidden-state collection for transformer models
- Works with DDP/FSDP/DeepSpeed and tensor parallelism
- Minimal dependencies (PyTorch only)

## Installation

```bash
pip install hookbox
```

Or from source:

```bash
pip install git+https://github.com/anonymous-authors/hookbox.git
```

## Quick Start

### Collecting Activations from Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hookbox import ActivationCollector

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hello, world!", return_tensors="pt")

collector = ActivationCollector(model)
with collector:
    data = collector.collect(inputs["input_ids"], inputs["attention_mask"])
    
    for layer_id, hidden in data.iter_layers():
        print(f"Layer {layer_id}: {hidden.shape}")

    layer_5 = data.get_layer(5)

    print(f"Logits: {data.logits.shape}")
```

### Custom Activation Capture

```python
from hookbox import HookManager, ActivationHook

model = ...

manager = HookManager(model)
activations = {}

count = manager.add_activation_hooks(
    predicate=lambda name, mod: "attn" in name and "c_proj" in name,
    on_activation=lambda x, name, mod: activations.update({name: x.clone()}),
)
print(f"Added {count} hooks")

output = model(input_ids)

for name, act in activations.items():
    print(f"{name}: {act.shape}")

manager.remove_all()
```

### Activation Patching / Interventions

```python
from hookbox import ActivationHook

def ablate_head(x, name, module):
    # x shape: [batch, seq, num_heads * head_dim]
    head_dim = x.shape[-1] // 12  # Assuming 12 heads
    x = x.clone()
    x[..., :head_dim] = 0
    return x

hook = ActivationHook(
    name="ablate_head_0",
    transform_fn=ablate_head,
)

hook.register(model.transformer.h[0].attn.c_proj)

output = model(input_ids)

hook.unregister()
```

## Distributed Models

hookbox handles distributed models transparently.

### DDP (DistributedDataParallel)

```python
from torch.nn.parallel import DistributedDataParallel as DDP

ddp_model = DDP(my_model)

manager = HookManager(ddp_model)
manager.add_hook("transformer.h.0.mlp", hook)  # Works!
```

### FSDP (FullyShardedDataParallel)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

fsdp_model = FSDP(my_model)

collector = ActivationCollector(fsdp_model)
with collector:
    data = collector.collect(input_ids)
```

### Tensor Parallelism

For tensor-parallel sharding, activations are per-rank shards. Use `gather_tensors` to combine them:

```python
from hookbox import HookManager, gather_tensors

manager = HookManager(tp_model)
activations = {}

def capture_and_gather(x, name, mod):
    full = gather_tensors(x, dim=-1)  # Gather along hidden dim
    activations[name] = full

manager.add_activation_hooks(
    predicate=lambda n, m: "mlp" in n,
    on_activation=capture_and_gather,
)
```

### Checking Distributed Status

```python
from hookbox import get_distributed_info, unwrap_model

info = get_distributed_info(model)
print(f"Wrapper: {info.wrapper_type}")
print(f"World size: {info.world_size}")
print(f"Rank: {info.rank}")
print(f"Is sharded: {info.is_sharded}")

base_model = unwrap_model(ddp_or_fsdp_model)
```

## Handling Activation Checkpointing

With gradient checkpointing, hooks fire twice (forward + recompute). By default the callback is skipped during recompute:

```python
hook = ActivationHook(
    name="capture",
    on_activation=callback,
    skip_checkpointing_recompute=True,  # Default
)

hook = ActivationHook(
    name="capture_all",
    on_activation=callback,
    skip_checkpointing_recompute=False,
)
```

## API Reference

### `ActivationCollector`

High-level interface for collecting activations from transformer models.

```python
collector = ActivationCollector(
    model,
    custom_hooks={
        "attn_weights": "transformer.h.0.attn",
    },
    gather_distributed=False,
    target_device=None,
)

with collector:
    data = collector.collect(input_ids, attention_mask)
    data = collector.collect_with_grad(input_ids)
```

### `CollectedActivations`

Container returned by `ActivationCollector.collect()`.

```python
data.hidden_states  # Tuple of tensors, one per layer (including embeddings)
data.logits         # Final output logits
data.custom         # Dict of custom activations

data.num_layers         # Number of transformer layers
data.embeddings         # Embedding layer output
data.last_hidden_state  # Final layer output
data.get_layer(5)       # Get specific layer
data.iter_layers()      # Iterate (layer_id, hidden_states) pairs

# Transformations (return new instances)
data.to("cuda")     # Move to device
data.cpu()          # Move to CPU
data.cuda(0)
data.detach()
data.clone()
data.half()
data.float()
```

### `HookManager`

Manages multiple hooks with automatic cleanup.

```python
manager = HookManager(model, auto_unwrap=True)

manager.add_activation_hooks(
    predicate=lambda name, mod: "layer" in name,
    on_activation=callback,
    transform_fn=transform,
    skip_checkpointing_recompute=True,
)

manager.add_gradient_hooks(
    predicate=lambda name, mod: "mlp" in name,
    on_gradient=callback,
)

manager.add_hook("path.to.module", hook)

manager.list_hooks()
manager.get_hook("name")
manager.num_hooks
manager.reset_all_counts()

manager.remove_hook("name")
manager.remove_all()

with HookManager(model) as manager:
    manager.add_activation_hooks(...)
    output = model(inputs)
```

### `ActivationHook`

Core hook for capturing/transforming activations.

```python
hook = ActivationHook(
    name="my_hook",
    on_activation=callback,
    transform_fn=transform,
    module_name="layer.0",
    skip_checkpointing_recompute=True,
)

hook.register(module)
hook.unregister()
hook.is_registered
hook.call_count
hook.reset_count()
```

### `GradientHook`

Hook for capturing/transforming gradients during backward pass.

```python
hook = GradientHook(
    name="grad_hook",
    on_gradient=callback,
    transform_fn=transform,
)
```

### `TensorHook`

Hook that attaches directly to a tensor for gradient capture.

```python
hook = TensorHook(
    name="hidden_grad",
    on_grad=lambda g: grads.append(g.clone()),
)
hook.register(tensor)
```

### Distributed Utilities

```python
from hookbox import (
    detect_wrapper_type,
    get_distributed_info,
    unwrap_model,
    get_module_by_name,
    gather_tensors,
    broadcast_tensor,
    DeviceMap,
    CheckpointingState,
    is_checkpointing_recomputation,
)
```

## Use Cases

- Mechanistic interpretability
- Activation patching
- Probing
- Feature steering
- Debugging
- Lens training

## Contributing

Contributions are welcome.

```bash
git clone https://github.com/anonymous-authors/hookbox.git
cd hookbox

pip install -e ".[dev]"

pytest tests/

pytest tests/ --cov=hookbox

mypy src/

ruff check src/
```