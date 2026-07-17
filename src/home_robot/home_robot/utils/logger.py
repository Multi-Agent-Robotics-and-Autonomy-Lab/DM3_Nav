import os
import logging

def create_run_log_dir(base_dir="logs"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while True:
        run_dir = os.path.join(base_dir, f"{run_id:04d}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        run_id += 1

def get_run_dir(base_dir="logs"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while True:
        run_dir = os.path.join(base_dir, f"{run_id:04d}")
        if not os.path.exists(run_dir):
            return run_id - 1
        run_id += 1

def get_logger(run_dir=None, agent_id=None):
    formatter = logging.Formatter(
        "[%(levelname)-5s] %(message)s", "%H:%M:%S"
    )
    if agent_id is None:
        logger = logging.getLogger("main")
        if not logger.handlers:
            if run_dir is None:
                run_dir = create_run_log_dir()
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(os.path.join(run_dir, "main.log"))
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger._run_dir = run_dir # store the run_dir in the logger for child loggers to access
        return logger

    logger = logging.getLogger(f"main.agent{agent_id}")
    if not logger.handlers:
        parent = logging.getLogger("main")
        logger.propagate = False  # don't write to parent's handlers
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(parent._run_dir, f"agent{agent_id}.log"))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
