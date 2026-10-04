"""OptChat: one endless chat whose history is the memory, as a Hermes ContextEngine plugin.

Install by copying or symlinking this directory to ``$HERMES_HOME/plugins/optchat`` and
activate explicitly with ``context: {engine: optchat}`` in config.yaml (never automatic).
This module exports exactly one ContextEngine subclass, ``OptChatEngine``, plus
``register(ctx)`` for the general plugin system (``ctx.register_context_engine``).
"""

from .engine import OptChatEngine

__all__ = ["OptChatEngine", "register"]


def register(ctx):
    ctx.register_context_engine(OptChatEngine())
