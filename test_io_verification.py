
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Agent_controller'))

def test_imports():
    try:
        from Agent_controller.Common.llm_interface import LLMInterface
        print("Success: LLMInterface imported.")
        
        from Agent_controller.Deliberate.io_agent import IOAgent
        print("Success: IOAgent imported.")
        
        # Test Instantiation
        llm = LLMInterface()
        serialized = llm.serialize_state({"test": 123})
        print(f"Serialization test: {serialized}")
        
        print("Verification Complete.")
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    test_imports()
