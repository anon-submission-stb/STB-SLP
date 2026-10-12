# coding: utf-8
import os
import shutil
import queue
import torch
import torch.nn as nn
import time
import pickle
import gzip
import numpy as np
from torch import Tensor
from batch import Batch
from torchtext.data import Dataset
from module.helpers import load_config, set_seed, load_checkpoint, log_cfg, make_model_dir, make_logger, ConfigurationError, get_latest_checkpoint, symlink_update
from data import load_data, make_data_iter
from model import build_model, Model
from torch.utils.tensorboard import SummaryWriter
from module.constants import TARGET_PAD
from loss import Loss
from builders import build_gradient_clipper, build_optimizer, build_scheduler
from plot_videos import plot_video, alter_DTW_timing
from prediction import validate_on_data

class TrainManager:

    def __init__(self, model: Model, config: dict, test=False, ckpt = None):

        train_config = config["training"]
        self.model_dir = train_config["model_dir"]

        # Build logger
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        # Build validation files
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir+"/tensorboard/")

        # model
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()
        self.target_pad = TARGET_PAD

        # New loss - depending on config
        self.loss = Loss(cfg = config, target_pad=self.target_pad)

        # normal
        self.normalization = "batch"

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)

        # TODO - Include Back Translation
        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ['bleu', 'chrf', "dtw"]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', 'DTW'")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                       "eval_metric")
        
        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in ["loss","dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', "
                                    "valid options: 'loss', 'dtw',.")
        
        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.last_epoch = -1
        self.batch_size = train_config["batch_size"]
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size",self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",self.batch_type)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        # comparison function for scores
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score
        
        ## Checkpoint restart
        # If continuing
        if ckpt:
            self.logger.info("Continuing model from %s", ckpt)
            self.init_from_checkpoint(ckpt)

        # Skip frames
        self.skip_frames = config["data"].get("skip_frames", 1)
        self.save_videos = config["training"].get("save_videos", False)

    # Save a checkpoint
    def _save_checkpoint(self, type="every") -> None:
        # Define model path
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        # Define State
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)
        # If this is the best checkpoint
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_best_queue.put(model_path)

            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update("{}_best.ckpt".format(self.steps), best_path)
            except OSError:
                # overwrite best.ckpt
                torch.save(state, best_path)

        # If this is just the checkpoint at every validation
        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_queue.put(model_path)

            every_path = "{}/every.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for the latest checkpoint
                symlink_update("{}_every.ckpt".format(self.steps), every_path)
            except OSError:
                # overwrite every.ckpt
                torch.save(state, every_path)

    # Initialise from a checkpoint
    def init_from_checkpoint(self, path: str) -> None:
        # Find last checkpoint
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None and \
                self.scheduler is not None:
            # Load the scheduler state
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        self.last_epoch = model_checkpoint["scheduler_state"]["last_epoch"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def _recalibrate_bn(self, train_data: Dataset, num_batches: int = 111) -> None:
        """Precise BN: reset BatchNorm running stats and re-estimate them as a cumulative average over
        num_batches training batches (train-mode forward, no grad, no weight update). RNG state is restored."""
        bns = [m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not bns:
            return
        momenta = [b.momentum for b in bns]
        for b in bns:
            b.reset_running_stats(); b.momentum = None
        was_training = self.model.training
        self.model.train()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.use_cuda else []), torch.no_grad():
            it = make_data_iter(train_data, batch_size=self.batch_size, batch_type=self.batch_type, train=True, shuffle=True)
            for i, tb in enumerate(it):
                if i >= num_batches:
                    break
                batch = Batch(torch_batch=tb, pad_index=self.pad_index, model=self.model)
                self.model.get_loss_for_batch(is_train=True, batch=batch, loss_function=self.loss)
        for b, mo in zip(bns, momenta):
            b.momentum = mo
        self.model.train(was_training)

    # Train and validate function
    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) -> None:

        # Make training iterator
        train_iter = make_data_iter(train_data,
                                    batch_size=self.batch_size,
                                    batch_type=self.batch_type,
                                    train=True, shuffle=self.shuffle)
        
        val_step = 0
        # Loop through epochs
        for epoch_no in range(self.last_epoch+1, self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            for batch in iter(train_iter):
                # reactivate training
                self.model.train()

                # create a Batch object from torchtext batch
                batch = Batch(torch_batch=batch,
                              pad_index=self.pad_index,
                              model=self.model)
                
                update = count == 0

                # Train the model on a batch
                batch_loss = self._train_batch(batch, update=update)

                self.tb_writer.add_scalar("train/train_batch_loss", batch_loss,self.steps)
                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()

                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:

                    valid_start_time = time.time()

                    # FGDM+STB: re-estimate BatchNorm running stats on train data before validation (precise BN);
                    # the stored momentum stats lag behind the drifting STB-output distribution
                    self._recalibrate_bn(train_data)

                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths = \
                        validate_on_data(
                            batch_size=self.eval_batch_size,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            model=self.model,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            batch_type=self.eval_batch_type,
                            type="val",
                        )
                    
                    val_step += 1

                    # Tensorboard writer
                    self.tb_writer.add_scalar("valid/valid_loss", valid_loss, self.steps)
                    self.tb_writer.add_scalar("valid/valid_score", valid_score, self.steps)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                        ckpt_score = valid_score
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        # Optionally render a few validation sequences
                        if self.save_videos:
                            display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                            self.produce_validation_video(
                                output_joints=valid_hypotheses,
                                inputs=valid_inputs,
                                references=valid_references,
                                model_dir=self.model_dir,
                                steps=self.steps,
                                display=display,
                                type="val_inf",
                                file_paths=valid_file_paths,
                            )

                    self._save_checkpoint(type="every")

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val",)

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f,  duration: %.4fs',
                            epoch_no+1, self.steps, valid_score,
                            valid_loss, valid_duration)

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                     self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no+1,
                             epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no+1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer


    # Produce the video of Phoenix MTC joints
    def produce_validation_video(self, output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None, dtw_file=None):

        # If not at test
        if type != "test":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")

        # If at test time
        elif type == "test":
            dir_name = model_dir + "/test_videos/"

        # Create model video folder if not exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # For sequence to display
        for i in display:

            seq = output_joints[i]
            ref_seq = references[i]
            input = inputs[i]
            # Write gloss label
            gloss_label = input[0]
            if input[1] != "</s>":
                gloss_label += "_" + input[1]
            if input[2] != "</s>":
                gloss_label += "_" + input[2]

            # Alter the dtw timing of the produced sequence, and collect the DTW score
            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)

            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))

            if file_paths is not None:
                sequence_ID = file_paths[i]
            else:
                sequence_ID = None

            print(sequence_ID + '    dtw: ' + '{0:.2f}'.format(float(dtw_score)))

            if dtw_file != None:
                dtw_file.writelines(sequence_ID + ' ' + '{0:.2f}'.format(float(dtw_score)) + '\n')

            # Plot this sequences video
            # if "<" not in video_ext:
            plot_video(joints=timing_hyp_seq,
                       file_path=dir_name,
                       video_name=video_ext,
                       references=ref_seq_count,
                       skip_frames=self.skip_frames,
                       sequence_ID=sequence_ID)


    # Save the generated skeletons (cut to the ground-truth length) as a gzipped pickle
    def save_skels(self, output_joints, references, display, model_dir, type, file_paths=None):
        out = []
        for i in display:
            ref = references[i][:, :-1]
            valid = (ref.abs().sum(-1) > 0).nonzero()
            n = int(valid[-1]) + 1 if len(valid) else ref.shape[0]
            out.append({'name': file_paths[i], 'sign': output_joints[i][:n, :-1].detach().cpu().float()})
        path = os.path.join(model_dir, "skels.%s.gz" % type)
        with gzip.open(path, "wb") as f:
            pickle.dump(out, f)
        print("Saved %d generated sequences to %s" % (len(out), path))

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:

        # Get loss from this batch
        batch_loss = self.model.get_loss_for_batch(is_train=True,
                                                          batch=batch,
                                                          loss_function=self.loss)

        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss = batch_loss / normalizer
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.backward()

        # FGDM+STB: global grad norm (before clipping); skip the update if it is > 10x the running median
        grads = [p_.grad.detach().norm() for p_ in self.model.parameters() if p_.grad is not None]
        gnorm = float(torch.norm(torch.stack(grads))) if grads else 0.0
        if not hasattr(self, '_gn_hist'):
            self._gn_hist, self._n_skipped = [], 0
        med = float(np.median(self._gn_hist)) if len(self._gn_hist) >= 50 else None
        skip = (not np.isfinite(gnorm)) or (med is not None and gnorm > 10 * med)
        if not skip:
            self._gn_hist = (self._gn_hist + [gnorm])[-200:]

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            if skip:
                self._n_skipped += 1
                self.logger.info("Skipped update at step %d: grad norm %.4g > 10 x median %.4g (skipped so far: %d)",
                                 self.steps, gnorm, med if med is not None else float('nan'), self._n_skipped)
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self.steps % self.logging_freq == 0 and med is not None:
                self.logger.info("Grad norm at step %d: %.4g (running median %.4g)", self.steps, gnorm, med)

            # increment step counter
            self.steps += 1

        # increment token counter
        self.total_tokens += batch.ntokens

        return norm_batch_loss

    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str, new_best: bool = False, report_type: str = "val") -> None:

        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))


    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params


def train(cfg_file: str, ckpt=None):

    # Load the config file
    cfg = load_config(cfg_file)

    # Set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    cfg["training"]["model_dir"] = make_model_dir(cfg["training"]["model_dir"],
                                        overwrite=cfg["training"].get("overwrite", False),test=False)

    # Load the data - Trg as (batch, # of frames, joints + 1 )
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

    # Build the FGDM model
    model = build_model(cfg=cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(config=cfg, model=model, test=False, ckpt=ckpt)

    # Store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
   
    shutil.copy2('./module/Diffusion.py', trainer.model_dir)
    shutil.copy2('./module/Denoiser.py', trainer.model_dir)
    shutil.copy2('./model.py', trainer.model_dir)
    # Log all entries of config
    log_cfg(cfg, trainer.logger)

    # Train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

def test(cfg_file, ckpt: str):

    # Load the config file
    cfg = load_config(cfg_file)
    cfg["training"]["model_dir"] = make_model_dir(cfg["training"]["model_dir"],
                                        overwrite=cfg["training"].get("overwrite", False),test=True)

    # Load the model directory and checkpoint
    model_dir = cfg["training"]["model_dir"]

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir,post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))
        
    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", True)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

    # Generate for the dev and test sets
    data_to_predict = {"dev": dev_data, "test": test_data}

    # Build the model; the checkpoint is loaded by the TrainManager below
    model = build_model(cfg=cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)

    # If cuda, set model as cuda
    if use_cuda:
        model.cuda()
    
    trainer = TrainManager(model=model, config=cfg, test=True, ckpt=ckpt)

    # FGDM+STB: precise BN before generation, as before every validation during training
    trainer._recalibrate_bn(train_data)

    # For each of the required data, produce results
    for data_set_name, data_set in data_to_predict.items():

        # Validate for this data set
        score, loss, references, hypotheses, inputs, all_dtw_scores, file_paths = \
            validate_on_data(
                model=model,
                data=data_set,
                batch_size=batch_size,
                max_output_length=max_output_length,
                eval_metric=eval_metric,
                loss_function=None,
                batch_type=batch_type,
                type="val" if data_set_name != "train" else "train_inf"
            )
        print('DTW score of %s set: %.3f' % (data_set_name, score))
        with open(os.path.join(model_dir, data_set_name + '_dtw.txt'), 'w') as f:
            f.write('DTW score of %s set: %.3f\n' % (data_set_name, score))

        display = list(range(len(hypotheses)))
        trainer.save_skels(output_joints=hypotheses, references=references, display=display, model_dir=model_dir,
                           type=data_set_name, file_paths=file_paths)

        # Optionally render a video for every generated sequence (slow)
        if cfg["training"].get("save_videos", False):
            trainer.produce_validation_video(output_joints=hypotheses, inputs=inputs, references=references,
                                             model_dir=model_dir, display=display, type="test",
                                             file_paths=file_paths)
