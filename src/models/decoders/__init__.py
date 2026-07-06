"""Output decoders."""

from .det_decoder import DetDecoder, nms
from .seg_decoder import SegDecoder

__all__ = ["DetDecoder", "SegDecoder", "nms"]
