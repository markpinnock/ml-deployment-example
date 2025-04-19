"""This is a simple example of how to train a model using Flax and JAX.

Based on: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""

import os
from functools import partial

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Use CPU to mimic multiple GPUs

import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from utils.sharding import MeshRules, named_sharding


class DotReluDot(nnx.Module):  # type: ignore[misc]
    """A simple model with two linear layers and a ReLU activation function."""

    def __init__(self, depth: int, rngs: nnx.Rngs, mesh_rules: MeshRules):
        """Initialize the model with the given depth and mesh rules.

        Args:
            depth: The depth of the model.
            rngs: The random number generator.
            mesh_rules: The mesh rules.
        """
        init_fn = nnx.initializers.lecun_normal()

        # Linear layer using nnx.Linear
        self.dot1 = nnx.Linear(
            depth,
            depth,
            kernel_init=nnx.with_partitioning(init_fn, mesh_rules("embed", "mlp")),
            use_bias=False,
            rngs=rngs,
        )

        # Linear layer directly using nnx.Param
        self.w2 = nnx.Param(
            init_fn(rngs.Params(), (depth, depth)), sharding=mesh_rules("mlp", "embed")
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Call the model with the given input.

        Args:
            x: The input data.

        Returns:
            The output data.
        """
        y = self.dot1(x)
        y = jax.nn.relu(y)
        y = jax.lax.with_sharding_constraint(y, PartitionSpec("data", "model"))

        return jnp.dot(y, self.w2.value)


@partial(nnx.jit, static_argnames=("hidden", "mesh_rules"))
def create_sharded_model(hidden: int, mesh_rules: MeshRules) -> nnx.Module:
    """Create a sharded model with the given hidden dimension and mesh rules.

    Args:
        hidden: The hidden dimension of the model.
        mesh_rules: The mesh rules.

    Returns:
        The sharded model.
    """
    model = DotReluDot(hidden, rngs=nnx.Rngs(0), mesh_rules=mesh_rules)
    # Shard model state
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


@nnx.jit  # type: ignore[misc]
def train_step(
    model: DotReluDot, optimiser: nnx.Optimizer, x: jax.Array, y: jax.Array
) -> jax.Array:
    """Train a single step of the model.

    Args:
        model: The model to train.
        optimiser: The optimizer.
        x: The input data.
        y: The target data.
    """

    def loss_fn(model: DotReluDot) -> jax.Array:
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimiser.update(grads)

    return loss


def main() -> None:
    """Main function to train the model."""
    # Setup mesh and mesh rules
    mesh = Mesh(
        devices=np.array(jax.devices()).reshape((2, 4)), axis_names=("data", "model")
    )
    mesh_rules = MeshRules(data="data", mlp="model")

    # Create sharded model and optimizer
    with mesh:
        model = create_sharded_model(hidden=1024, mesh_rules=mesh_rules)
        optimiser = nnx.Optimizer(model, optax.adam(1e-3))

    # Create input and label data and shard them
    data_sharding = named_sharding(mesh, "data", None)
    input_ = jax.device_put(
        jax.random.normal(jax.random.key(0), (8, 1024)), data_sharding
    )
    label = jax.device_put(
        jax.random.normal(jax.random.key(1), (8, 1024)), data_sharding
    )

    # Train the model
    with mesh:
        for i in range(5):
            loss = train_step(model, optimiser, input_, label)
            print(i, loss)


if __name__ == "__main__":
    main()
