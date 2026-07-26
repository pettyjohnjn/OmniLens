# demo.py
import torch
import torch.nn as nn

# Add src to path if not installed
import sys; sys.path.insert(0, "src")

from hookbox import HookManager, ActivationHook, ActivationCollector

# Simple model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(3)])
        self.head = nn.Linear(32, 100)
    
    def forward(self, x, output_hidden_states=False, **kwargs):
        h = self.embed(x)
        hidden = [h] if output_hidden_states else None
        for layer in self.layers:
            h = torch.relu(layer(h))
            if output_hidden_states:
                hidden.append(h)
        logits = self.head(h)
        
        class Out:
            pass
        out = Out()
        out.logits = logits
        out.hidden_states = tuple(hidden) if hidden else None
        return out

model = TinyModel()
input_ids = torch.randint(0, 100, (2, 8))

# Demo 1: HookManager with predicate
print("=== Demo 1: HookManager ===")
activations = {}

manager = HookManager(model)
count = manager.add_activation_hooks(
    predicate=lambda name, mod: name.startswith("layers."),
    on_activation=lambda x, name, mod: activations.update({name: x.shape}),
)
print(f"Added {count} hooks")

_ = model(input_ids)
for name, shape in activations.items():
    print(f"  {name}: {shape}")

manager.remove_all()

# Demo 2: ActivationCollector
print("\n=== Demo 2: ActivationCollector ===")
collector = ActivationCollector(model)
with collector:
    data = collector.collect(input_ids)

print(f"Collected {data.num_layers} layers")
for layer_id, hidden in data.iter_layers():
    print(f"  Layer {layer_id}: {hidden.shape}")

# Demo 3: Transform activations
print("\n=== Demo 3: Intervention (zero out layer 1) ===")
baseline = model(input_ids).logits.clone()

hook = ActivationHook(
    name="zero_layer1",
    transform_fn=lambda x, n, m: x * 0,  # Zero out!
)
hook.register(model.layers[1])

intervened = model(input_ids).logits
hook.unregister()

print(f"Baseline output[0,0,:5]:   {baseline[0,0,:5].tolist()}")
print(f"Intervened output[0,0,:5]: {intervened[0,0,:5].tolist()}")
print(f"Outputs differ: {not torch.allclose(baseline, intervened)}")