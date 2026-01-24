from message import Message, MessageLevel
from optimizer_agent import OptimizerAgent


class DummyIOAgent:
    """Minimal agent that sends one optimization request."""

    def __init__(self, name="io_agent"):
        self.name = name

    def send(self, target_agent):
        payload = {
            "pv": [0.0] * 24,
            "demand": [2.0] * 24,
            "price": [
                0.20, 0.20, 0.20, 0.20,
                0.25, 0.30, 0.35, 0.40,
                0.45, 0.50, 0.50, 0.45,
                0.40, 0.35, 0.30, 0.25,
                0.20, 0.20, 0.20, 0.20,
                0.20, 0.20, 0.20, 0.20,
            ],
            "battery": {
                "capacity_kwh": 10.0,
                "max_charge_kw": 3.0,
                "max_discharge_kw": 3.0,
                "initial_soc": 5.0,
                "min_soc": 1.0,
                "max_soc": 10.0,
            },
        }

        msg = Message(
            level=MessageLevel.REQUEST,
            msg_type="optimization_request",
            payload=payload,
            sender=self.name,
            target=target_agent.name,
        )

        return target_agent.handle_message(msg, meta={})


if __name__ == "__main__":
    optimizer = OptimizerAgent()
    io_agent = DummyIOAgent()

    reply = io_agent.send(optimizer)

    if reply is None:
        print("No reply received")
    elif reply.level == MessageLevel.ERROR:
        print("ERROR:", reply.payload)
    else:
        print("Optimization result:")
        print("Total cost:", reply.payload["total_cost"])
        print("Battery power (kW):", reply.payload["battery_schedule"]["power_kw"])
        print("Battery SoC:", reply.payload["battery_schedule"]["soc"])
        print("Grid power (kW):", reply.payload["grid_schedule"]["power_kw"])
