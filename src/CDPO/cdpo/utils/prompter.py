"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./src/CDPO/cdpo/templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        prompt: str,
        chosen: Union[None, str] = None,
        rejected: Union[None, str] = None,
    ) -> str:
        
        prompt = self.template["prompt_no_input"].format(
            instruction=prompt
        )
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()
