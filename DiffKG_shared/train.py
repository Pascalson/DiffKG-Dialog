import os
import math
import logging
from pprint import pformat

import torch
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage, Average
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)

from utils import *#make_logdir, get_dataset, get_shared_knowledge_graph
from models import *#DiffkgT5

from argparse import ArgumentParser
import time

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

def set_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="runs", help="Path to save the model")
    parser.add_argument("--model_type", type=str, default="diffkg-t5", help="Short name of the model to train")
    parser.add_argument("--max_hops_num", type=int, default=3, help="Maximum number of hops in reasoning")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()
    logger.info("Arguments: %s", pformat(args))
    return args


def train():
    args = set_args()

    logger.info("Prepare models")

    # Set number of relations
    KG = get_shared_knowledge_graph(os.path.join(args.data_dir,"KG"))
    args.N_r = len(KG['rels_dict']) 
    N_E = len(KG["ents_dict"])

    # Load model
    model = DiffkgT5(args).to(args.device)

    # load dialogues with labeled reasoning path
    train_set = get_dataset(model.tokenizer, args.data_dir, "train", KG)
    train_loader = model.get_dataloader(train_set, args.train_batch_size, N_E)
    valid_set = get_dataset(model.tokenizer, args.data_dir, "valid", KG)
    valid_loader = model.get_dataloader(valid_set, args.valid_batch_size, N_E)
    
    # transform knowledge graph into sparse metrices
    KG_sparse_Ms = get_KG_sparse_metrices(KG, args.device)

    # tokenize entities
    ents_map = get_entities_mappings(list(KG["ents_dict"].keys()), model.tokenizer, model._model.config.vocab_size).to(args.device)
    model.build_entities_embeddings(ents_map)

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)

        rel_loss, ent_loss, outputs = model(batch, KG_sparse_Ms, labels=batch)
        loss = outputs.loss
        loss /= args.gradient_accumulation_steps
        rel_loss /= args.gradient_accumulation_steps
        ent_loss /= args.gradient_accumulation_steps
        
        loss.backward(retain_graph=True)
        rel_loss.backward(retain_graph=True)
        ent_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return ((loss.item() + rel_loss.item() + ent_loss.item()) * args.gradient_accumulation_steps,)

    def inference(engine, batch):
        model.eval()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        with torch.no_grad():
            rel_loss, ent_loss, outputs = model(batch, KG_sparse_Ms, labels=batch)
        return (rel_loss.item() + ent_loss.item() + outputs.loss.item(),)


    trainer = Engine(update)
    evaluator = Engine(inference)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(valid_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(valid_loader))


    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    Average(output_transform=lambda x: x[0]).attach(evaluator, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

    log_dir = make_logdir(args.output_dir)

    # checkpoints
    checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=args.n_epochs)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})
    torch.save(args, log_dir + '/model_training_args.bin')
    model.tokenizer.save_pretrained(log_dir)

    # save losses
    train_loss_logger = []
    valid_loss_logger = []

    def save_losses():
        # save losses
        torch.save({'train':train_loss_logger,'valid':valid_loss_logger}, \
            os.path.join(log_dir, 'losses.bin'))

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_trainer_logger(engine):
        train_loss_logger.append(engine.state.metrics.copy())
        save_losses()

    @evaluator.on(Events.COMPLETED)
    def save_valid_logger(engine):
        valid_loss_logger.append(engine.state.metrics.copy())
        save_losses()

    trainer.run(train_loader, max_epochs=args.n_epochs)

    if args.n_epochs > 0:
        os.rename(checkpoint_handler.last_checkpoint, os.path.join(log_dir, WEIGHTS_NAME))

if __name__ == "__main__":
    train()
