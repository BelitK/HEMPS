 
FORBIDDEN_NAME_PARTS = [
        "create_agent",
        "connect_agent",
        "set_agent",
        "repeat_step",
        "step_",
        "plan",
        "task",
        "do_",
        ]
class CheckTools:
    def __init__(self):
        pass
    @staticmethod
    def unique_name(self, base: str, existing: set[str]) -> str:
        """Unique name guard."""
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"
    @staticmethod
    def reject_bad_name(name: str) -> None:
        """Reject bad agent names."""
        lowered = name.lower()
        for bad in FORBIDDEN_NAME_PARTS:
            if bad in lowered:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid agent name '{name}'. Use a domain noun like house_battery, pv_panels, ev_charger."
                    ),
                )
                