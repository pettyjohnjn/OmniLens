# tests/test_hooks.py
"""Tests for hookbox."""

import pytest
import torch
import torch.nn as nn

from hookbox import (
    ActivationCollector,
    ActivationHook,
    BaseHook,
    CollectedActivations,
    GradientHook,
    HookManager,
    TensorHook,
    WrapperType,
    detect_wrapper_type,
    get_distributed_info,
    unwrap_model,
)

# =============================================================================
# Test Fixtures
# =============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size: int = 8, num_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, 100)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                use_cache=False, return_dict=True):
        x = self.embed(input_ids)
        hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = torch.relu(layer(x))
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.head(x)

        if return_dict:
            class Output:
                pass
            out = Output()
            out.logits = logits
            out.hidden_states = tuple(hidden_states) if hidden_states else None
            return out
        return logits


class MockDDP(nn.Module):
    """Mock DDP wrapper for testing."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(hidden_size=8, num_layers=3)


@pytest.fixture
def input_ids():
    """Create sample input."""
    return torch.randint(0, 100, (2, 4))


# =============================================================================
# BaseHook Tests
# =============================================================================

class TestBaseHook:
    """Tests for BaseHook."""

    def test_abstract_class(self):
        """Test that BaseHook cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseHook("test")

    def test_subclass_implementation(self):
        """Test that subclasses can be created."""
        class MyHook(BaseHook):
            def register(self, module):
                self._handle = module.register_forward_hook(lambda m, i, o: o)
                self.module = module

        hook = MyHook("test")
        assert hook.name == "test"
        assert not hook.is_registered


# =============================================================================
# ActivationHook Tests
# =============================================================================

class TestActivationHook:
    """Tests for ActivationHook."""

    def test_capture_activations(self, simple_model, input_ids):
        """Test capturing activations."""
        captured = []
        hook = ActivationHook(
            name="test",
            on_activation=lambda x, n, m: captured.append(x.clone()),
        )

        hook.register(simple_model.layers[0])
        assert hook.is_registered

        _ = simple_model(input_ids)

        assert len(captured) == 1
        assert captured[0].shape == (2, 4, 8)
        assert hook.call_count == 1

        hook.unregister()
        assert not hook.is_registered

    def test_transform_activations(self, simple_model, input_ids):
        """Test transforming activations."""
        scale_factor = 2.0
        hook = ActivationHook(
            name="scale",
            transform_fn=lambda x, n, m: x * scale_factor,
        )

        # Get baseline output
        baseline = simple_model(input_ids).logits.clone()

        # Apply hook and compare
        hook.register(simple_model.layers[0])
        scaled = simple_model(input_ids).logits

        # Output should be different due to scaling
        assert not torch.allclose(baseline, scaled)

        hook.unregister()

        # Should return to baseline
        restored = simple_model(input_ids).logits
        assert torch.allclose(baseline, restored)

    def test_capture_and_transform(self, simple_model, input_ids):
        """Test both capturing and transforming."""
        captured = []
        hook = ActivationHook(
            name="both",
            on_activation=lambda x, n, m: captured.append(x.clone()),
            transform_fn=lambda x, n, m: x * 0,  # Zero out
        )

        hook.register(simple_model.layers[0])
        _ = simple_model(input_ids)

        # Should have captured
        assert len(captured) == 1
        # Captured value should NOT be transformed (captured before transform)
        assert captured[0].abs().sum() > 0

        hook.unregister()

    def test_context_manager(self, simple_model, input_ids):
        """Test using hook as context manager."""
        captured = []
        hook = ActivationHook(
            name="ctx",
            on_activation=lambda x, n, m: captured.append(x.clone()),
        )

        hook.register(simple_model.layers[0])
        with hook:
            _ = simple_model(input_ids)

        # Hook should be unregistered after context
        assert not hook.is_registered
        assert len(captured) == 1

    def test_call_count_and_reset(self, simple_model, input_ids):
        """Test call counting and reset."""
        hook = ActivationHook(name="counter")
        hook.register(simple_model.layers[0])

        assert hook.call_count == 0

        _ = simple_model(input_ids)
        assert hook.call_count == 1

        _ = simple_model(input_ids)
        assert hook.call_count == 2

        hook.reset_count()
        assert hook.call_count == 0

        hook.unregister()

    def test_repr(self):
        """Test string representation."""
        hook = ActivationHook(
            name="test",
            on_activation=lambda x, n, m: None,
            module_name="layer.0",
        )
        repr_str = repr(hook)
        assert "test" in repr_str
        assert "layer.0" in repr_str
        assert "captures=True" in repr_str


# =============================================================================
# GradientHook Tests
# =============================================================================

class TestGradientHook:
    """Tests for GradientHook."""

    def test_capture_gradients(self, simple_model, input_ids):
        """Test capturing gradients."""
        gradients = []
        hook = GradientHook(
            name="grad",
            on_gradient=lambda g, n, m: gradients.append(g.clone()),
        )

        hook.register(simple_model.layers[0])

        output = simple_model(input_ids).logits
        loss = output.sum()
        loss.backward()

        assert len(gradients) == 1
        assert hook.call_count == 1

        hook.unregister()


# =============================================================================
# TensorHook Tests
# =============================================================================

class TestTensorHook:
    """Tests for TensorHook."""

    def test_capture_tensor_gradient(self):
        """Test capturing gradient on a tensor."""
        grads = []
        hook = TensorHook(
            name="tensor_grad",
            on_grad=lambda g: grads.append(g.clone()),
        )

        x = torch.randn(3, 4, requires_grad=True)
        hook.register(x)

        y = (x ** 2).sum()
        y.backward()

        assert len(grads) == 1
        assert grads[0].shape == x.shape

        hook.unregister()

    def test_requires_grad_check(self):
        """Test that registering on non-grad tensor raises."""
        hook = TensorHook(name="test")
        x = torch.randn(3, 4, requires_grad=False)

        with pytest.raises(ValueError, match="requires_grad"):
            hook.register(x)


# =============================================================================
# HookManager Tests
# =============================================================================

class TestHookManager:
    """Tests for HookManager."""

    def test_add_hook_by_name(self, simple_model, input_ids):
        """Test adding a hook by module name."""
        manager = HookManager(simple_model)
        captured = []

        hook = ActivationHook(
            name="layer0",
            on_activation=lambda x, n, m: captured.append(x.clone()),
        )

        manager.add_hook("layers.0", hook)
        assert manager.num_hooks == 1

        _ = simple_model(input_ids)
        assert len(captured) == 1

        manager.remove_all()
        assert manager.num_hooks == 0

    def test_add_hook_invalid_module(self, simple_model):
        """Test adding hook to non-existent module."""
        manager = HookManager(simple_model)
        hook = ActivationHook(name="test")

        with pytest.raises(ValueError, match="not found"):
            manager.add_hook("nonexistent.module", hook)

    def test_add_hooks_by_predicate(self, simple_model, input_ids):
        """Test adding hooks by predicate."""
        manager = HookManager(simple_model)
        captured = {}

        def capture(x, name, mod):
            captured[name] = x.clone()

        count = manager.add_activation_hooks(
            predicate=lambda name, mod: name.startswith("layers."),
            on_activation=capture,
        )

        assert count == 3  # 3 layers

        _ = simple_model(input_ids)
        assert len(captured) == 3

        manager.remove_all()

    def test_remove_specific_hook(self, simple_model):
        """Test removing a specific hook."""
        manager = HookManager(simple_model)

        hook1 = ActivationHook(name="hook1")
        hook2 = ActivationHook(name="hook2")

        manager.add_hook("layers.0", hook1)
        manager.add_hook("layers.1", hook2)

        assert manager.num_hooks == 2

        removed = manager.remove_hook("hook1")
        assert removed
        assert manager.num_hooks == 1
        assert "hook2" in manager
        assert "hook1" not in manager

    def test_list_hooks(self, simple_model):
        """Test listing hooks."""
        manager = HookManager(simple_model)

        manager.add_hook("layers.0", ActivationHook(name="a"))
        manager.add_hook("layers.1", ActivationHook(name="b"))

        hooks = manager.list_hooks()
        assert set(hooks) == {"a", "b"}

    def test_context_manager(self, simple_model, input_ids):
        """Test manager as context manager."""
        captured = []

        with HookManager(simple_model) as manager:
            manager.add_activation_hooks(
                predicate=lambda n, m: "layers" in n,
                on_activation=lambda x, n, m: captured.append(n),
            )
            _ = simple_model(input_ids)

        # Hooks should be removed after context
        assert len(captured) == 3

    def test_duplicate_hook_name_raises(self, simple_model):
        """Test that duplicate hook names raise error."""
        manager = HookManager(simple_model)

        manager.add_hook("layers.0", ActivationHook(name="dup"))

        with pytest.raises(ValueError, match="already registered"):
            manager.add_hook("layers.1", ActivationHook(name="dup"))

    def test_reset_all_counts(self, simple_model, input_ids):
        """Test resetting all hook call counts."""
        manager = HookManager(simple_model)

        manager.add_activation_hooks(
            predicate=lambda n, m: "layers" in n,
            on_activation=lambda x, n, m: None,
        )

        _ = simple_model(input_ids)

        # All hooks should have been called
        for _, hook in manager.iter_hooks():
            assert hook.call_count == 1

        manager.reset_all_counts()

        for _, hook in manager.iter_hooks():
            assert hook.call_count == 0

        manager.remove_all()


# =============================================================================
# Distributed Tests
# =============================================================================

class TestDistributed:
    """Tests for distributed utilities."""

    def test_unwrap_model_plain(self, simple_model):
        """Test unwrap on plain model (no-op)."""
        unwrapped = unwrap_model(simple_model)
        assert unwrapped is simple_model

    def test_unwrap_model_ddp(self, simple_model):
        """Test unwrap on DDP-wrapped model."""
        wrapped = MockDDP(simple_model)
        unwrapped = unwrap_model(wrapped)
        assert unwrapped is simple_model

    def test_unwrap_nested(self, simple_model):
        """Test unwrap with nested wrappers."""
        wrapped = MockDDP(MockDDP(simple_model))
        unwrapped = unwrap_model(wrapped)
        assert unwrapped is simple_model

    def test_detect_wrapper_type_none(self, simple_model):
        """Test detecting no wrapper."""
        wrapper_type = detect_wrapper_type(simple_model)
        assert wrapper_type == WrapperType.NONE

    def test_detect_wrapper_type_unknown(self, simple_model):
        """Test detecting unknown wrapper (has .module but not DDP)."""
        wrapped = MockDDP(simple_model)
        wrapper_type = detect_wrapper_type(wrapped)
        assert wrapper_type == WrapperType.UNKNOWN

    def test_get_distributed_info(self, simple_model):
        """Test getting distributed info."""
        info = get_distributed_info(simple_model)
        assert info.wrapper_type == WrapperType.NONE
        assert info.world_size == 1
        assert info.rank == 0
        assert not info.is_distributed
        assert info.is_main_process

    def test_hook_manager_with_wrapped_model(self, simple_model, input_ids):
        """Test HookManager with wrapped model."""
        wrapped = MockDDP(simple_model)
        manager = HookManager(wrapped)

        captured = []
        manager.add_activation_hooks(
            predicate=lambda n, m: n == "layers.0",
            on_activation=lambda x, n, m: captured.append(x.clone()),
        )

        # Should work through the wrapper
        _ = wrapped(input_ids)
        assert len(captured) == 1

        manager.remove_all()


# =============================================================================
# ActivationCollector Tests
# =============================================================================

class TestActivationCollector:
    """Tests for ActivationCollector."""

    def test_collect_hidden_states(self, simple_model, input_ids):
        """Test collecting hidden states."""
        collector = ActivationCollector(simple_model)

        with collector:
            data = collector.collect(input_ids)

        assert data.num_layers == 3
        assert len(data.hidden_states) == 4  # embed + 3 layers
        assert data.logits is not None

    def test_iter_layers(self, simple_model, input_ids):
        """Test iterating over layers."""
        collector = ActivationCollector(simple_model)

        with collector:
            data = collector.collect(input_ids)

        layers = list(data.iter_layers())
        assert len(layers) == 3

        for layer_id, hidden in layers:
            assert hidden.shape == (2, 4, 8)

    def test_get_layer(self, simple_model, input_ids):
        """Test getting specific layer."""
        collector = ActivationCollector(simple_model)

        with collector:
            data = collector.collect(input_ids)

        layer_1 = data.get_layer(1)
        assert layer_1.shape == (2, 4, 8)

        # Out of range should raise
        with pytest.raises(IndexError):
            data.get_layer(10)

    def test_custom_hooks(self, simple_model, input_ids):
        """Test custom hooks."""
        collector = ActivationCollector(
            simple_model,
            custom_hooks={
                "head_input": "head",
            }
        )

        with collector:
            data = collector.collect(input_ids)

        assert "head_input" in data.custom
        assert data.custom["head_input"].shape == (2, 4, 8)

    def test_collect_with_grad(self, simple_model, input_ids):
        """Test collecting with gradients."""
        collector = ActivationCollector(simple_model)

        with collector:
            data = collector.collect_with_grad(input_ids)

        # Should be able to backprop through hidden states
        loss = data.logits.sum()
        loss.backward()  # Should not raise

    def test_is_attached_property(self, simple_model):
        """Test is_attached property."""
        collector = ActivationCollector(simple_model)

        assert not collector.is_attached

        collector.attach()
        assert collector.is_attached

        collector.detach()
        assert not collector.is_attached

    def test_with_wrapped_model(self, simple_model, input_ids):
        """Test collector with wrapped model."""
        wrapped = MockDDP(simple_model)
        collector = ActivationCollector(wrapped)

        with collector:
            data = collector.collect(input_ids)

        assert data.num_layers == 3


# =============================================================================
# CollectedActivations Tests
# =============================================================================

class TestCollectedActivations:
    """Tests for CollectedActivations."""

    def test_to_device(self):
        """Test moving to device."""
        data = CollectedActivations(
            hidden_states=(torch.randn(2, 4, 8), torch.randn(2, 4, 8)),
            logits=torch.randn(2, 4, 100),
            custom={"test": torch.randn(2, 4, 8)},
        )

        # Should not raise
        data_cpu = data.to("cpu")
        assert data_cpu.hidden_states[0].device.type == "cpu"

    def test_detach(self):
        """Test detaching from graph."""
        x = torch.randn(2, 4, 8, requires_grad=True)
        data = CollectedActivations(
            hidden_states=(x,),
            logits=None,
        )

        detached = data.detach()
        assert not detached.hidden_states[0].requires_grad

    def test_clone(self):
        """Test cloning."""
        original = torch.randn(2, 4, 8)
        data = CollectedActivations(
            hidden_states=(original,),
        )

        cloned = data.clone()

        # Should be equal but different objects
        assert torch.allclose(cloned.hidden_states[0], original)
        cloned.hidden_states[0][0, 0, 0] = 999
        assert not torch.allclose(cloned.hidden_states[0], original)

    def test_embeddings_property(self):
        """Test embeddings property."""
        embed = torch.randn(2, 4, 8)
        layer1 = torch.randn(2, 4, 8)

        data = CollectedActivations(
            hidden_states=(embed, layer1),
        )

        assert torch.equal(data.embeddings, embed)

    def test_last_hidden_state_property(self):
        """Test last_hidden_state property."""
        embed = torch.randn(2, 4, 8)
        layer1 = torch.randn(2, 4, 8)
        layer2 = torch.randn(2, 4, 8)

        data = CollectedActivations(
            hidden_states=(embed, layer1, layer2),
        )

        assert torch.equal(data.last_hidden_state, layer2)

    def test_half_and_float(self):
        """Test dtype conversions."""
        data = CollectedActivations(
            hidden_states=(torch.randn(2, 4, 8),),
            logits=torch.randn(2, 4, 100),
        )

        half_data = data.half()
        assert half_data.hidden_states[0].dtype == torch.float16

        float_data = half_data.float()
        assert float_data.hidden_states[0].dtype == torch.float32

    def test_repr(self):
        """Test string representation."""
        data = CollectedActivations(
            hidden_states=(torch.randn(2, 4, 8),) * 4,
            logits=torch.randn(2, 4, 100),
            custom={"test": torch.randn(2, 4, 8)},
        )

        repr_str = repr(data)
        assert "num_layers=3" in repr_str
        assert "has_logits=True" in repr_str
        assert "test" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self, simple_model, input_ids):
        """Test a complete workflow."""
        # Collect baseline
        collector = ActivationCollector(simple_model)
        with collector:
            baseline = collector.collect(input_ids)

        # Apply intervention
        manager = HookManager(simple_model)

        def zero_layer(x, name, mod):
            return x * 0

        manager.add_activation_hooks(
            predicate=lambda n, m: n == "layers.1",
            transform_fn=zero_layer,
        )

        # Collect with intervention
        with collector:
            intervened = collector.collect(input_ids)

        manager.remove_all()

        # Outputs should be different due to intervention
        assert not torch.allclose(
            baseline.logits,
            intervened.logits
        )

    def test_capture_specific_and_all(self, simple_model, input_ids):
        """Test capturing both specific modules and all hidden states."""
        collector = ActivationCollector(
            simple_model,
            custom_hooks={
                "layer_0_out": "layers.0",
                "layer_2_out": "layers.2",
            }
        )

        with collector:
            data = collector.collect(input_ids)

        # Should have all hidden states
        assert data.num_layers == 3

        # Should also have custom captures
        assert "layer_0_out" in data.custom
        assert "layer_2_out" in data.custom

        # Custom captures should match corresponding hidden states
        # (layer 0 output is hidden_states[1], layer 2 output is hidden_states[3])
        assert torch.allclose(data.custom["layer_0_out"], data.get_layer(0))
        assert torch.allclose(data.custom["layer_2_out"], data.get_layer(2))
