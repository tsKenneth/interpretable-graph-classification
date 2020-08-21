from os.path import dirname, basename, isfile, join
import glob

# Import new models here
from .DGCNN import DGCNN
from .DiffPool import DiffPool
from .DiffPoolD import DiffPoolD
from .GCN import GCN
from .GCND import GCND

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
