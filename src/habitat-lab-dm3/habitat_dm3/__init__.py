"""DM3-Nav's additions to habitat-lab; importing this package registers them with habitat."""
from habitat_dm3.config.default_structured_configs import CONFIG_DIR
from habitat_dm3.datasets.goat import goat_dataset  # noqa: F401
from habitat_dm3.sims.habitat_simulator import habitat_simulator  # noqa: F401
from habitat_dm3.tasks.nav import goat_task, nav, object_nav_task  # noqa: F401
