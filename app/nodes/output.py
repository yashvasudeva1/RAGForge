"""
Answer output node — kept here per the scaffold convention.

The actual implementation lives in ``app.nodes.input`` alongside QueryInputNode.
This module re-exports AnswerOutputNode to honour the file layout.
"""

from app.nodes.input import AnswerOutputNode

__all__ = ["AnswerOutputNode"]
