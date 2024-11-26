import logging
import os
import torch
from models.base import ChatBase

class CodeBase(ChatBase):
    def __init__(
        self,
        model="mixtral-8x7b-instruct",
        use_local=False,
        temperature=0.3,
        peft_model="",
    ):
        super().__init__(model, use_local, temperature, peft_model)

    @property
    def system_prompt(self):
        return """You are a helpful coding assistant. You will follow the user's instructions to complete the task."""