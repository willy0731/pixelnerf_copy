import os.path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tqdm
import warnings

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, args, conf, device=None):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.train_dataset_loader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False
        )
        self.test_dataset_loader = DataLoader(
            test_dataset,
            batch_size=min(args.batch_size, 16),
            shuffle=True,
            num_workers=4,
            pin_memory=False
        )
        self.num_total_batches = len(self.train_dataset) # 3619728 for llff
        self.exp_ame = args.name
        self.save_interval = conf.get_int("save_interval")
        self.print_interval = conf.get_int("print_interval")
        self.vis_interval = conf.get_int("vis_interval")
        self.eval_interval = conf.get_int("eval_interval")
        self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1)
        self.num_epochs = args.epochs
        self.accu_grad = conf.get_int("accu_grad", 1)
        self.summary_path = os.path.join(args.logs_path, args.name)
        self.writer = SummaryWriter(self.summary_path)
        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test
        os.makedirs(self.summary_path, exist_ok=True) # logs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.gamma != 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optmizer = self.optimizer, gamma = args.gamma
            )
        else:
            self.lr_scheduler = None
        # Load weights
        # self.managed_weight_saving = hasattr(model, "load_weights")
        # if self.managed_weight_saving:
        #     model.load_weights(self.args)
        # 還沒打完
        self.start_iter_id = 0
    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def train_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None
    
    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + str(losses[k]) for k in losses))
        
        def data_loop(dl):
            while True:
                for x in iter(dl):
                    yield x
        
        test_data_iter = data_loop(self.test_dataset_loader)
        step_id = self.start_iter_id
        progress = tqdm.tqdm(bar_format="[{bar}] {n}/{total} [{elapsed}<{remaining}]")
        for epoch in range(self.num_epoch_repeats):
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], global_step=step_id)
            
            batch = 0
            for _ in range(self.num_epoch_repeats):
                for data in self.train_dataset_loader:
                    losses = self.train_step(data, global_step=step_id)