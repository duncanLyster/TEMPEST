from .factory import TemperatureSolverFactory
from .tempest_standard import TempestStandardSolver

# Register available solvers
TemperatureSolverFactory.register("tempest_standard", TempestStandardSolver) 