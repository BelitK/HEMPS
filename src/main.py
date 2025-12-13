"""
Do not modify!
"""
import asyncio
import mango

from env import UserAgent, UserResponse, create_test_env, create_test_user_requests, Environment

class ExampleAgent(mango.Agent):

    def __init__(self, environment: Environment):
        super().__init__()
        
        self.env = environment

    def on_ready(self):
        # calculate a sensible schedule for the battery, for that purpose you need some information from the environment
        battery_kw = self.env.house_information.battery.power_kw
        battery_capa = self.env.house_information.battery.size_kwh
        # this is only the base forecast, the added power usage announced by the user needs to be integrated to that
        demand_forecast = self.env.house_information.demand
        # information about the devices the user might wanna use
        load_devices = self.env.house_information.load_devices
        # the power schedule forecast of the solar power plant
        solar_power_plant_forecast = self.env.house_information.solar_power_plant.forecast
        # the dynamic consumption rate per kWh (15min res as all forecasts), you can assume that all generated energy above the demand will be sold
        # at this rate, while below it will be bought at this rate
        price_forecast = self.env.price

    def handle_message(self, content, meta):
        print(f"Got request {content} from user agent!")
        self.schedule_instant_message(UserResponse("Not a good explanation, but something at least."), mango.sender_addr(meta))


async def main():
    container = mango.create_tcp_container("127.0.0.1:8883")
    
    environment = create_test_env()

    hems_coordinator_agent = container.register(ExampleAgent(environment)) # change to your coordinator agent!

    # the user agent needs always to be in the container to send the requests, to do some testing its a good
    # idea to use different user requests 
    user_agent = container.register(UserAgent(hems_coordinator_agent.addr, create_test_user_requests()))

    async with mango.activate(container):
        await user_agent.done.wait()
        

if __name__ == "__main__":
    asyncio.run(main())
