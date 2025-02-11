class TemperatureSolverFactory:
    _solvers = {}

    @classmethod
    def register(cls, name, solver_class):
        cls._solvers[name] = solver_class

    @classmethod
    def create(cls, solver_name):
        if solver_name not in cls._solvers:
            raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {list(cls._solvers.keys())}")
        return cls._solvers[solver_name]()

    @classmethod
    def validate_parameters(cls, solver_name, config):
        solver = cls._solvers[solver_name]()
        required_params = solver.get_required_parameters()
        missing_params = [param for param in required_params if not hasattr(config, param)]
        if missing_params:
            raise ValueError(f"Missing required parameters for {solver_name}: {missing_params}") 