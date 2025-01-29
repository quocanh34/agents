from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterable, Generic, Literal, TypeVar, Union

from livekit import rtc

from .. import llm


@dataclass
class InputSpeechStartedEvent:
    pass


@dataclass
class InputSpeechStoppedEvent:
    pass


@dataclass
class MessageGeneration:
    message_id: str
    text_stream: AsyncIterable[str]
    audio_stream: AsyncIterable[rtc.AudioFrame]


@dataclass
class GenerationCreatedEvent:
    message_stream: AsyncIterable[MessageGeneration]
    function_stream: AsyncIterable[llm.FunctionCall]


@dataclass
class ErrorEvent:
    type: str
    message: str


@dataclass
class RealtimeCapabilities:
    message_truncation: bool
    input_audio_sample_rate: int


class RealtimeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RealtimeModel:
    def __init__(self, *, capabilities: RealtimeCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> RealtimeCapabilities:
        return self._capabilities

    @abstractmethod
    def session(self) -> "RealtimeSession": ...

    @abstractmethod
    async def aclose(self) -> None: ...


EventTypes = Literal[
    "input_speech_started",  # serverside VAD (also used for interruptions)
    "input_speech_stopped",  # serverside VAD
    "generation_created",
    "error",
]

TEvent = TypeVar("TEvent")


class RealtimeSession(
    ABC,
    rtc.EventEmitter[Union[EventTypes, TEvent]],
    Generic[TEvent],
):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__()
        self._realtime_model = realtime_model

    @property
    def realtime_model(self) -> RealtimeModel:
        return self._realtime_model

    @property
    @abstractmethod
    def chat_ctx(self) -> llm.ChatContext: ...

    @property
    @abstractmethod
    def fnc_ctx(self) -> llm.FunctionContext: ...

    @abstractmethod
    async def update_instructions(self, instructions: str) -> None: ...

    @abstractmethod
    async def update_chat_ctx(
        self, chat_ctx: llm.ChatContext
    ) -> None: ...  # can raise RealtimeError on Timeout

    @abstractmethod
    async def update_fnc_ctx(
        self, fnc_ctx: llm.FunctionContext | list[llm.AIFunction]
    ) -> None: ...

    @abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    @abstractmethod
    def generate_reply(
        self,
    ) -> asyncio.Future[
        GenerationCreatedEvent
    ]: ...  # can raise RealtimeError on Timeout

    # cancel the current generation (do nothing if no generation is in progress)
    @abstractmethod
    def interrupt(self) -> None: ...

    # message_id is the ID of the message to truncate (inside the ChatCtx)
    @abstractmethod
    def truncate(self, *, message_id: str, audio_end_ms: int) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...
