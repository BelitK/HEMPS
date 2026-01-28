from mango import create_tcp_container, create_topology, JSON
from agents import dynamic_agent, optimizer_agent
from agents.message import Message, MessageLevel


codec = JSON()
codec.add_serializer(*Message.__serializer__())
container = create_tcp_container(("127.0.0.1", 0))
topology_ctx = create_topology()
topology = topology_ctx.__enter__()

agent_1 = optimizer_agent.OptimizerAgent()

# ...existing code...
container.register(agent_1)
# create a Message instance and encode it (use agent_1.addr for receiver)
msg = Message(MessageLevel.REQUEST, 'test optimization and message', container.addr, agent_1.addr)
mess = codec.encode(msg)
container.send_message(content=mess, receiver_addr=agent_1.addr)
# ...existing code...