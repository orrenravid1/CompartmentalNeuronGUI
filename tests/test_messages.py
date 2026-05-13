from dataclasses import dataclass

import pytest

from compneurovis.messages import (
    SET_CONTROL,
    STATUS,
    CommandPayload,
    MessageType,
    Reset,
    SetControl,
    Status,
    command_message,
    message_type_for_payload,
    update_message,
)


def test_command_message_infers_registered_message_type():
    message = command_message(SetControl("gain", 0.5))

    assert message.intent == "command"
    assert message.type is SET_CONTROL
    assert message.type.name == "set_control"
    assert message.payload == SetControl("gain", 0.5)
    assert message_type_for_payload(message.payload) is SET_CONTROL


def test_update_message_infers_registered_message_type():
    message = update_message(Status("ready", 100))

    assert message.intent == "update"
    assert message.type is STATUS
    assert message.payload == Status("ready", 100)


def test_message_type_rejects_disallowed_intent():
    with pytest.raises(ValueError, match="does not allow 'command' intent"):
        command_message(Status("not a command"), message_type=STATUS)


def test_message_type_rejects_wrong_payload_class():
    with pytest.raises(TypeError, match="expects payload SetControl"):
        command_message(Reset(), message_type=SET_CONTROL)


def test_custom_payloads_require_explicit_message_type():
    @dataclass(frozen=True, slots=True)
    class CustomCommand(CommandPayload):
        value: int

    payload = CustomCommand(1)

    with pytest.raises(ValueError, match="No registered message type"):
        command_message(payload)

    custom_type = MessageType("custom_command", CustomCommand, ("command",))
    message = command_message(payload, message_type=custom_type)

    assert message.type is custom_type
    assert message.payload is payload
