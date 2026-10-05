"""OptChat: one endless chat whose history is the memory, as a Hermes ContextEngine plugin.

Install by copying or symlinking this directory to ``$HERMES_HOME/plugins/optchat`` and
activate explicitly with ``context: {engine: optchat}`` in config.yaml (never automatic).
This module exports exactly one ContextEngine subclass, ``OptChatEngine``, plus
``register(ctx)`` for the general plugin system (``ctx.register_context_engine``).
"""

from .engine import OptChatEngine
from .summarizer import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_TIMEOUT, TASK

__all__ = ["OptChatEngine", "register"]


def register(ctx):
    # The directory context-engine collector has only register_context_engine; the
    # general plugin context additionally offers an owned auxiliary task slot.
    if hasattr(ctx, "register_auxiliary_task"):
        ctx.register_auxiliary_task(
            TASK, display_name="OptChat compactor", description="Summarize OptChat tree nodes",
            defaults={"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL,
                      "timeout": DEFAULT_TIMEOUT},
        )
    ctx.register_context_engine(OptChatEngine())
