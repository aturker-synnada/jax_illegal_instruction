import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def example_fn():
    x = jnp.array([1, 2, 3], dtype=jnp.bfloat16)
    y = jnp.array([1.0, 2, 3], dtype=jnp.bfloat16)
    jnp.allclose(x, y, 1e-5, 1e-5)

    print("example_fn done")


if __name__ == "__main__":
    example_fn()