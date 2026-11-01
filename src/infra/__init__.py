"""Colab operations: session setup, detached jobs, status, result sync. Knows nothing about robots or policies.

The colab-mcp tools act through the browser tab and are only available to the agent, so the logic lives here and the
agent or notebook calls it from one-line cells, e.g.  !python -m infra status
"""
