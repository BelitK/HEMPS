
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Agent_controller'))

def test_imports():
    try:
        from Agent_controller.Deliberate.io_agent import IOAgent
        print("Success: IOAgent imported.")
        
        # Test Instantiation
        agent = IOAgent(dispatch_agent_addr="test_addr")
        print(f"Instantiation test: {agent.system_context}")
        
        # Test Validation
        is_safe = agent.validate_input("Hello")
        is_unsafe = agent.validate_input("Infinite power")
        print(f"Validation test: Safe={is_safe}, Unsafe={not is_unsafe}")

        print("Verification Complete.")
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    test_imports()
