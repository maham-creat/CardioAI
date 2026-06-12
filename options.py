import argparse

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # Universal Relative Paths
        self.parser.add_argument('--raw_data_path', type=str, default="./data/raw_dataset", help='Relative path to the raw dataset folder')
        self.parser.add_argument('--output_dir', type=str, default="./processed_dataset", help='Relative target directory for output matrices')
        self.parser.add_argument('--seed', type=int, default=42, help='Global execution random state seed')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Input training batch size')
        self.parser.add_argument('--epochs', type=int, default=30, help='Total training passes over the dataset')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate factor for optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay rate coefficient')
        self.parser.add_argument('--model_dir', type=str, default="./saved_models", help='Relative target folder for model binaries')
        self.parser.add_argument('--num_workers', type=int, default=2, help='Subprocess data loading workers')

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Inference evaluation batch size')
        self.parser.add_argument('--threshold', type=float, default=0.5, help='Classification probability decision threshold')
        self.parser.add_argument('--model_dir', type=str, default="./saved_models", help='Relative folder housing saved weights')
        self.parser.add_argument('--test_output_dir', type=str, default="./test_metrics_output", help='Target directory for validation metrics logs')
        self.parser.add_argument('--num_workers', type=int, default=2, help='Subprocess loading workers')