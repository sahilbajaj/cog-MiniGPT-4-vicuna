from cog import BasePredictor, Input
from minigpt4.common.config import Config
import argparse


class Predictor(BasePredictor):
    def setup(self):
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        args = parser.parse_args(["--cfg-path","eval_configs/minigpt4_eval.yaml"])

        self.cfg = Config(args)
        self.prefix = "hello"

    def predict(self, text: str = Input(description="Text to prefix with 'hello '")) -> str:
        return self.prefix + " " + text
