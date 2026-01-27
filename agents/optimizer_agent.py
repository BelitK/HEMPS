from typing import Any, Dict, List

from .dynamic_agent import DynamicAgent
from .message import Message, MessageLevel
from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)

# TODO change the schedule. communicate only with scheduler. update at the beginning of the day
# TODO look into the scheduler and understand how it forwards information
# TODO http requests to get info from scheduler
# TODO fix message format for the communication with the scheduler
# TODO use @http.localhost.80000/scheduler/@_schedule or something like that, add payload to request with necessary info from optimization result
class OptimizerAgent(DynamicAgent):
    """
    Optimizer Agent

    Responsibilities:
    1. Receive forecasts + battery parameters from IO agent
    2. Optimize battery + grid schedule
    3. Reply with optimal schedule and total cost
    """

    TYPE = "optimizer"
    LABEL = "Optimizer Agent"
    DEFAULT_PERSONA = "Optimizes battery and grid operation to minimize energy cost."
    DEFAULT_USAGE = "Receives forecasts and returns optimal schedules."
    CAPABILITIES = ["optimize", "schedule", "cost"]

    def __init__(
        self,
        name: str = "optimizer_agent",
        persona: str | None = None,
        usage: str | None = None,
    ):
        super().__init__(
            name=name,
            persona=persona or self.DEFAULT_PERSONA,
            usage=usage or self.DEFAULT_USAGE,
        )

    # ------------------------
    # Core optimization logic
    # ------------------------

    def optimize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pv: List[float] = payload["pv"]
        demand: List[float] = payload["demand"]
        price: List[float] = payload["price"]
        battery: Dict[str, float] = payload["battery"]

        T = range(24)

        # Battery parameters
        capacity = battery["capacity_kwh"]
        max_charge = battery["max_charge_kw"]
        max_discharge = battery["max_discharge_kw"]
        soc_init = battery["initial_soc"]
        soc_min = battery["min_soc"]
        soc_max = battery["max_soc"]

        # LP problem
        prob = LpProblem("Energy_Optimization", LpMinimize)

        # Decision variables
        battery_power = LpVariable.dicts(
            "battery_power",
            T,
            lowBound=-max_charge,
            upBound=max_discharge,
        )

        grid_power = LpVariable.dicts(
            "grid_power",
            T,
            lowBound=None,   # allow export
            upBound=None,
        )

        soc = LpVariable.dicts(
            "soc",
            range(25),
            lowBound=soc_min,
            upBound=soc_max,
        )

        # Initial SoC
        prob += soc[0] == soc_init

        # Constraints
        for t in T:
            # Energy balance
            prob += (
                pv[t] + battery_power[t] + grid_power[t]
                == demand[t]
            )

            # SoC dynamics (1h timestep)
            prob += soc[t + 1] == soc[t] - battery_power[t]

        # Objective: minimize cost
        prob += lpSum(price[t] * grid_power[t] for t in T)

        # Solve
        status = prob.solve()

        if LpStatus[status] != "Optimal":
            raise RuntimeError(f"Optimization failed: {LpStatus[status]}")

        # Extract results
        battery_schedule = [value(battery_power[t]) for t in T]
        soc_schedule = [value(soc[t]) for t in T]
        grid_schedule = [value(grid_power[t]) for t in T]
        total_cost = value(prob.objective)

        return {
            "battery_schedule": {
                "power_kw": battery_schedule,
                "soc": soc_schedule,
            },
            "grid_schedule": {
                "power_kw": grid_schedule,
            },
            "total_cost": round(total_cost, 2),
        }

    # ------------------------
    # Message handling
    # ------------------------

    def handle_message(self, content, meta):
        if not isinstance(content, Message):
            return super().handle_message(content, meta)

        # Only react to optimization requests
        if content.level != MessageLevel.REQUEST:
            return

        try:
            result = self.optimize(content.payload)

            reply = Message(
                level=MessageLevel.REPLY,
                msg_type="optimization_result",
                payload=result,
                sender=self.name,
                target=content.sender,
            )

            return reply

        except Exception as e:
            error_msg = Message(
                level=MessageLevel.ERROR,
                msg_type="optimization_error",
                payload={"error": str(e)},
                sender=self.name,
                target=content.sender,
            )
            return error_msg



