from dataclasses import dataclass

from jax.sharding import Mesh, NamedSharding, PartitionSpec


def named_sharding(mesh: Mesh, *names: str | None) -> NamedSharding:
    """Create a named sharding for a given mesh and names.

    Args:
        mesh: Mesh to create the named sharding for
        names: Axis names to create the named sharding for

    Returns:
        Named sharding for the given mesh and names
    """
    return NamedSharding(mesh, PartitionSpec(*names))


@dataclass(unsafe_hash=True)
class MeshRules:
    """Dataclass for sharding rules.

    Notes:
        Taken from: https://huggingface.co/blog/jiagaoxiang/jax-nnx-sharding

    Attributes:
        embed: Sharding rule for embedding-like dimensions
        mlp: Sharding rule for MLP layers dimensions
        data: Sharding rule for data batch dimension
    """

    embed: str | None = None
    mlp: str | None = None
    data: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        """Get sharding rules for given keys."""
        return tuple(getattr(self, key) for key in keys)
