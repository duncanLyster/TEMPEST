from .factory import TemperatureSolverFactory
from .tempest_standard import TempestStandardSolver
from .tempest_implicit import TempestImplicitSolver

# Register available solvers
TemperatureSolverFactory.register("tempest_standard", TempestStandardSolver)
TemperatureSolverFactory.register("tempest_implicit", TempestImplicitSolver)
