import struct
from enum import IntEnum


PROTOCOL_MAGIC = b"LSIM"
PROTOCOL_VERSION = 1
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct("!4sHHQ")
HEADER_SIZE = _HEADER.size


class ProtocolError(RuntimeError):
    pass


class MessageType(IntEnum):
    HELLO = 1
    REGISTER_TEMPLATE = 2
    TEMPLATE_READY = 3
    RUN_BATCH = 4
    RUN_WAVE = 5
    BATCH_DONE = 6
    PASS = 7
    SLEEP = 8
    EXIT = 9
    ERROR = 10
    FILE_WORKLOAD = 11
    BATCH_ACCEPTED = 12
    ADVANCE_TIME = 13


def encode_frame(message_type, payload=b"", version=PROTOCOL_VERSION):
    try:
        message_type = MessageType(message_type)
    except ValueError as error:
        raise ProtocolError(f"Unknown message type: {message_type}") from error
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Protocol payload must be bytes-like.")
    payload = bytes(payload)
    return _HEADER.pack(
        PROTOCOL_MAGIC,
        int(version),
        int(message_type),
        len(payload),
    ) + payload


def decode_header(header, expected_version=PROTOCOL_VERSION,
                  max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES):
    if len(header) != HEADER_SIZE:
        raise ProtocolError(
            f"Invalid header size: expected {HEADER_SIZE}, got {len(header)}")
    magic, version, raw_type, payload_size = _HEADER.unpack(header)
    if magic != PROTOCOL_MAGIC:
        raise ProtocolError(f"Invalid protocol magic: {magic!r}")
    if version != expected_version:
        raise ProtocolError(
            f"Unsupported protocol version: expected {expected_version}, got {version}")
    try:
        message_type = MessageType(raw_type)
    except ValueError as error:
        raise ProtocolError(f"Unknown message type: {raw_type}") from error
    if payload_size > max_payload_bytes:
        raise ProtocolError(
            f"Payload size {payload_size} exceeds limit {max_payload_bytes}")
    return message_type, payload_size


def _receive_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ProtocolError(
                f"Unexpected EOF with {remaining} protocol bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(connection, expected_version=PROTOCOL_VERSION,
                  max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES):
    header = _receive_exact(connection, HEADER_SIZE)
    message_type, payload_size = decode_header(
        header,
        expected_version=expected_version,
        max_payload_bytes=max_payload_bytes,
    )
    return message_type, _receive_exact(connection, payload_size)


def send_frame(connection, message_type, payload=b"",
               version=PROTOCOL_VERSION):
    frame = encode_frame(message_type, payload, version=version)
    connection.sendall(frame)
    return len(frame)
