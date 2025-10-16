import importlib.metadata
import logging

__version__ = importlib.metadata.version("cs336-systems")

_root = logging.getLogger()
if not _root.handlers:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
elif _root.level > logging.INFO:
	_root.setLevel(logging.INFO)
