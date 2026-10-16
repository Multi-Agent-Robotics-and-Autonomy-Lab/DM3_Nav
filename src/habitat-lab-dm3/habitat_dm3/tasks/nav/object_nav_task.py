"""Multi-agent ObjectNav task."""
from habitat.core.registry import registry
from habitat_dm3.tasks.nav.nav import MultiAgentNavigationTask


@registry.register_task(name="MultiAgentObjectNav-v1")
class MultiAgentObjectNavigationTask(MultiAgentNavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
