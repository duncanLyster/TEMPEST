from .factory import TemperatureSolverFactory
from .tempest_standard import TempestStandardSolver
from .thermprojrs import ThermprojrsSolver

# Register available solvers
TemperatureSolverFactory.register("tempest_standard", TempestStandardSolver)
TemperatureSolverFactory.register("thermprojrs", ThermprojrsSolver) 