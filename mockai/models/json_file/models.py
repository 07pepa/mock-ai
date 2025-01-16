from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal, TypeAlias

from mockai.models.api.anthropic import Payload as AnthropicPayload
from mockai.models.api.openai import Message
from mockai.models.api.openai import Payload as OpenAIPayload
from mockai.models.common import FunctionOutput, FunctionOutputs

ResponseType: TypeAlias = Literal["text", "function"]


class InputMatcher(BaseModel):
    system_prompt_name: Annotated[str | None, Field(default=None)]
    role: Annotated[str | None, Field(default=None)]
    content: Annotated[str | None, Field(default=None)]
    offset: Annotated[
        int | None, Field(default=None)
    ]  # always matches no mater the match mode if none search for match in all messages
    match_mode: Annotated[Literal["ANY", "ALL"], Field(default="ALL")]

    def _content_match(self, content: str | list[str]) -> bool:
        if self.content is None:
            return False
        if self.content is not None and isinstance(content, list):
            for one in content:
                if self.content == one.text:
                    return True
                return False
        return self.content == content

    def is_matching_payload(
        self, payload: AnthropicPayload | OpenAIPayload, system_prompts: dict[str, str]
    ) -> bool:
        if self.offset is not None:
            msg_count = len(payload.messages)
            if not -msg_count <= self.offset < msg_count:
                return False
            messages = [payload.messages[self.offset]]
        else:
            messages = payload.messages

        system_prompt = None
        if (
            isinstance(payload, AnthropicPayload)
            and self.system_prompt_name is not None
        ):
            system_prompt = system_prompts[self.system_prompt_name]

        aggregator = any if self.match_mode == "ANY" else all
        for message in messages:
            if system_prompt:
                arr = [system_prompt == payload.system]
            else:
                arr = []
            arr.extend(
                [
                    self._content_match(message.content),
                    self.role == message.role,
                ]
            )

            if aggregator(arr):
                return True
        return False


class PreDeterminedResponse(BaseModel):
    type: ResponseType
    input: InputMatcher | str
    output: str | FunctionOutput | FunctionOutputs

    @model_validator(mode="after")
    def verify_structure(self) -> PreDeterminedResponse:
        if self.type == "function":
            try:
                if isinstance(self.output, list):
                    checks = [isinstance(f, FunctionOutput) for f in self.output]
                    assert all(checks)
                else:
                    assert isinstance(self.output, FunctionOutput)
            except AssertionError:
                raise ValueError(
                    "When a response is of type 'function', the output must be a single FunctionOutput object or an array of FunctionOutput objects"
                )
        elif self.type == "text":
            if isinstance(self.output, str) is False:
                raise ValueError(
                    "When a response is of type 'text', the output must be a string."
                )
        return self

    def response_matches(
        self, payload: AnthropicPayload | OpenAIPayload, system_prompts: dict[str, str]
    ) -> bool:
        if isinstance(self.input, str):
            return self.input == payload.messages[-1].content
        else:
            return self.input.is_matching_payload(payload, system_prompts)


class PreDeterminedResponses(BaseModel):
    responses: Annotated[list[PreDeterminedResponse], Field(default=[])]
    system_prompts: Annotated[dict[str, str], Field(default={})]

    @model_validator(mode="after")
    def _verify_responses(self) -> PreDeterminedResponses:
        glob_errors = []
        temp_errors = []
        for response in self.responses:
            if (
                isinstance(response.input, InputMatcher)
                and response.input.system_prompt_name is not None
                and response.input.system_prompt_name not in self.system_prompts
            ):
                temp_errors.append(response.input.system_prompt_name)
        if len(temp_errors) > 0:
            glob_errors.append(
                f"Following system prompt missing for the following system_prompt_names: {temp_errors}"
            )
            temp_errors = []

        if len(glob_errors) > 0:
            raise ValueError(
                "found following errors in file:\n" + "\n".join(glob_errors)
            )
        return self

    def __iter__(self):  # type: ignore
        return iter(self.responses)

    def __setitem__(self, idx, item):
        self.responses[idx] = item

    def pop(self, idx):
        self.responses.pop(idx)

    def append(self, response: PreDeterminedResponse):
        self.responses.append(response)

    def find_matching_or_none(
        self, payload: AnthropicPayload | OpenAIPayload
    ) -> PreDeterminedResponse | None:
        for response in self.responses:
            if response.response_matches(payload, self.system_prompts):
                return response
        return None
