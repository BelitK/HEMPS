
# Mini-Project: Explainable HEMS

In the explainable HEMS you need to develop an agent/multi-agent system (MAS) using `mango-agents`, which has the task to manage a household with a solar panel, a battery and the ability to profit from dynamic tariffs. The goal for the agent(s) is to minimize the cost for one day (24 hours, 15min resolution) and include users' consumption information. Further it should be able to explain the decisions it makes to the home owner in a natural language. Here, the agent represents a household with a PV plant and battery storage. This implies the agent system can:

* Compute a cost minimizing schedule of the battery
* Include simple power consumption information provided by the user formulated in a natural language (e.g. "I want to use my washing machine at 11am.")
* Explain the chosen/calculated schedule to the user in a natural language

The agent system works in a given environment and can observe:

* The device information (solar and battery)
* The demand and price forecast for one day

Further, it can receive two different kinds of user requests:

1. Request for explanation, the answer needs to be in natural language
2. An information affecting the forecasted schedule. These requests always imply some additional power usage that should be integrated into the planning process. There is a limited set of devices which will be referenced in that way, the necessary information on these devices is available.

Some general assumptions you can make:

* Assume a single dynamic tariff: the same price applies for buying from and selling to the grid
* You may assume that user requests are grammatically simple and unambiguous, and refer only to known devices.
* The scheduling can be done centrally, e.g. a linear programming approach to calculate the schedule would be sufficient.

Note: We recommend to have some knowledge regarding the handling of (large) language models beforehand. Further you need have appropriate hardware to run language models locally (smaller LLMs around 7B-20B should be sufficient, e.g. gpt-oss-20b).

## The Environment

The environment in this project is given as part of the project files. For generating explanations, you should use a language model. For an example how to use language models within an mango-agent, see e.g. https://github.com/OFFIS-DAI/mango/compare/development...feature-llm-agents.

## Evaluation

We will evaluate your project results in a freshly set up environment and only include your agent system. 

## Grading

There are multiple requirements for passing:

1. Presentation of your project results at the final discussion (all members of the group need to be present!)
2. Your project is adequately documented. We expect a short description of your agent and its strategies and, depending on your implementation, unique usage characteristics for your project. (-> README is sufficient)
3. Your project fulfills the project goals adequately
    * Your agent system works in the provided environment (~= creates reasonably cost optimized schedules, communicates with the user)
    * Your agent system generates fitting answers to the user request in a natural language (~= for passing, explanations must mention the key drivers of the schedule, e.g., low prices, high demand, solar availability, or user-specified consumption)
    * Your agent system does not rely on the specific values of the parameters used by the provided environment  (i.e., the specific battery parameters, the forecasts, etc.).

To receive the grade-improving bonus, you need to hit one of the following marks:

* Your agent system is outstanding and of excellent quality (code and methodological!)
* Your agent system clearly explains all decisions, and can reliably integrate user constraints
* If you have an interesting idea to extend the scope of the mini project, discuss this with us, and for successful implementation, you will also receive the grade-improving bonus