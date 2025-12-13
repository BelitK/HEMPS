from dataclasses import dataclass
import numpy as np
from enum import Enum
import random
import time

import mango
import asyncio

T = 96
DEMAND_FORECAST = -np.random.random(T) * 10
SOLAR_FORECAST = np.random.random(T) * 5
PRICE_FORECAST = np.random.random(T) * 5

@dataclass
class Battery:
    size_kwh: float
    power_kw: float

@dataclass
class SolarPowerPlant:
    forecast: np.ndarray

@dataclass
class Device:
    p_max: float
    p_min: float
    description: str

@dataclass
class HouseInformation:
    battery: Battery 
    solar_power_plant: SolarPowerPlant
    load_devices: list[Device] 
    demand: np.ndarray

class RequestType(Enum):
    INFORM = 0
    EXPLAIN = 1

@dataclass
class UserRequest:
    type: RequestType
    message: str

@dataclass
class UserResponse:
    message: str

@dataclass
class Environment:
    house_information: HouseInformation
    price: np.ndarray


def create_test_user_requests():
    return [UserRequest(RequestType.EXPLAIN, "Please explain the schedule"), 
            UserRequest(RequestType.INFORM, "I will use the washing machine at 12am."),
            UserRequest(RequestType.EXPLAIN, "How did you adjust the schedule regarding the washing machine information")]

def create_test_env():
    return Environment(HouseInformation(Battery(10, 5), 
                                        SolarPowerPlant(SOLAR_FORECAST), 
                                        load_devices=[Device(3, 1, "Wasching machine"), Device(5, 3, "EV")], 
                                        demand=DEMAND_FORECAST),
                       price=PRICE_FORECAST)


class UserAgent(mango.Agent):

    def __init__(self, communicate_to: mango.AgentAddress, user_requests: list[UserRequest]):
        super().__init__()

        self.partner_addr = communicate_to

        # ordered user requests 
        self.user_requests = user_requests
        self.done = asyncio.Event()

    def on_ready(self):
        # send first user request
        self.schedule_instant_message(self.user_requests[0], self.partner_addr)
    
    def handle_message(self, content, meta):
        if type(content) == UserResponse:
            print(f"The response to the last request is: {content.message}, this was a response " + \
                  f"to the request ({self.user_requests[0].type}) {self.user_requests[0].message}")

            # current request is done, go to next or finish
            self.user_requests.pop(0)

            if len(self.user_requests) > 0:
                # send the next message in random() / 10 seconds
                self.schedule_timestamp_task(self.send_message(self.user_requests[0], self.partner_addr), time.time() + random.random() / 10)
            else:
                # all request were sent and all answers have been received.
                self.done.set()