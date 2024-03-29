import argparse
import random

import torch
from torchvision import datasets, transforms

from evaluators.base import PredictionBasedEvaluator
from metrics.accuracy import Accuracy
from models.cifar_10_models.vgg import VGG7
from trainers.base import TrainerConfig
from trainers.pretraining import Trainer
from utilities.utils import xavier_uniform_initialize, gpu_information_summary, set_seed

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

cifar_transform_valid = transforms.Compose([transforms.ToTensor(), normalize])
cifar_transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        normalize,
    ]
)


def main(args):
    n_gpu, _ = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)
    cifar_data_train = datasets.CIFAR10(root=args.data_root, train=True, transform=cifar_transform_train, download=True)
    if args.memorize:
        random.shuffle(cifar_data_train.targets)
    cifar_data_valid = datasets.CIFAR10(
        root=args.data_root, train=False, transform=cifar_transform_valid, download=True
    )
    model = VGG7()
    model.apply(xavier_uniform_initialize)
    training_config = TrainerConfig(
        prediction_evaluator=PredictionBasedEvaluator(metrics=[Accuracy()]),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        seed_value=args.seed_value,
        nb_epochs=40,
        num_workers=10,
        batch_size=128,
        logging_step=100,
        loggers=None,
        log_dir=None,
        verbose=True,
        save_progress=True,
        saving_dir="model_zoo",
        use_scheduler=True,
        nb_warmup_steps=0,
        learning_rate=1e-3,
        experiment_name=args.experiment_name,
        nb_classes=10,
        max_grad_norm=1.0,
        progress_history=1,
    )
    trainer = Trainer(
        model=model, train_dataset=cifar_data_train, valid_dataset=cifar_data_valid, config=training_config
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memorize",
        help="Whether to train the network to memorize.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--data_root",
        help="Path to the root data.",
        type=str,
    )

    parser.add_argument(
        "--seed_value",
        help="Random seed value for this experiment.",
        type=int,
        default=3407,
    )
    parser.add_argument("--experiment_name", help="Optional name for the experiment.", type=str, default=None)

    args = parser.parse_args()
    main(args=args)
