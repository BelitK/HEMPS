from enum import Enum
from typing import List, Optional, TypeAlias, Enum 


# define payload as type
Payload: TypeAlias = dict[List[float]]


class MessageLevel(Enum):
    # info, action, warning, error, critical
    # info, request, command, action, error, critical 
    # decide?

    INFO = 'info'
    REQUEST = 'request'
    COMMAND = 'command'
    ACTION = 'action'
    ERROR = 'error'
    WARNING = 'warning'
    CRITICAL = 'critical'



class Message:
    """
    Generic multi-agent message.
    Payload is opaque to the serializer.
    """

    def __init__(
        self,
        level: MessageLevel,
        msg_type: str,
        payload: Payload,
        sender: Optional[str] = None,
        target: Optional[str] = None,
    ) -> None:
        self.level: MessageLevel = level
        self.msg_type: str = msg_type
        self.payload: Payload = payload
        self.sender: Optional[str] = sender
        self.target: Optional[str] = target

    def __asdict__(self) -> dict:
        return {
            "level": self.level.value,
            "type": self.msg_type,
            "payload": self.payload,
            "sender": self.sender,
            "target": self.target,
        }

    @classmethod
    def __fromdict__(cls, attrs: dict) -> "Message":
        attrs = attrs.copy()
        attrs["level"] = MessageLevel(attrs["level"])
        return cls(**attrs)

    @classmethod
    def __serializer__(cls):
        return (cls, cls.__asdict__, cls.__fromdict__)
