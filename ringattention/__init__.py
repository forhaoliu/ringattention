"""This module contains RingAttention forward and backward pass, supporting both blockwise computation in Jax and fused computation in Pallas.
It features blockwise computation for feedforward networks to reduce memory cost.
For more details, refer to RingAttention ('Ring Attention with Blockwise Transformers for Near-Infinite Context') at https://arxiv.org/abs/2310.01889 and BlockwiseTransformer ('Blockwise Parallel Transformer for Large Context Models') at https://arxiv.org/abs/2305.19370.
"""

from .ringattention_inference import ring_attention_inference
from .ringattention_jax import ring_attention as ringattention_jax
from .ringattention_pallas_tpu import ring_flash_attention_tpu as ringattention_pallas_tpu
import jax

__all__ = ["ring_attention_inference", "ringattention_jax", "ringattention_pallas_tpu"]
