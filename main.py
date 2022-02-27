import argparse

import torch

from evaluators.base import PredictionBasedEvaluator
from losses.cka_map_loss import CKAMapLossCE
from metrics.accuracy import Accuracy
from models.cifar10_models import VGG
from torchvision import datasets, transforms

from target_maps.comical import GOBLIN
from trainers.base import TrainerConfig
from trainers.maptraining import MapTrainingConfig
import trainers
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
    cifar_data_valid = datasets.CIFAR10(
        root=args.data_root, train=False, transform=cifar_transform_valid, download=True
    )
    model = VGG()
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    else:
        model.apply(xavier_uniform_initialize)

    if not args.with_map:
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
            experiment_name="pretraining_model_on_cifar10",
            nb_classes=10,
            max_grad_norm=1.0,
            progress_history=1,
        )
        trainer = Trainer(
            model=model, train_dataset=cifar_data_train, valid_dataset=cifar_data_valid, config=training_config
        )
        trainer.train()
    else:
        if args.experiment_name == "GoblinCKA":
            target_cka = GOBLIN
        else:
            raise Exception("Experiment name provided is not supported.", args.experiment_name)
        training_config = MapTrainingConfig(
            prediction_evaluator=PredictionBasedEvaluator(metrics=[Accuracy()]),
            optimizer=None,
            seed_value=args.seed_value,
            nb_epochs=50,
            num_workers=10,
            batch_size=128,
            logging_step=100,
            loggers=None,
            log_dir=None,
            verbose=True,
            save_progress=True,
            saving_dir="model_zoo",
            use_scheduler=True,
            nb_warmup_steps=500,
            learning_rate=1e-3,
            experiment_name=args.experiment_name,
            nb_classes=10,
            max_grad_norm=1.0,
            progress_history=1,
            cka_alpha=20.0,
            cka_difference_function="LogCosh",
            target_cka=target_cka,
            criterion=CKAMapLossCE(alpha=20.0, mse=False),
        )
        trainer = trainers.maptraining.Trainer(
            model=model, train_dataset=cifar_data_train, valid_dataset=cifar_data_valid, config=training_config
        )
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with_map",
        help="Whether to train with a CKA map objective or not.",
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
    parser.add_argument(
        "--model_path",
        help="Path of the model the to load/continue training from.",
        type=str,
    )
    parser.add_argument("--experiment_name", help="Optional name for the experiment.", type=str, default=None)
    parser.add_argument(
        "--cka_path",
        help="Path to the cka map of the a reference model.",
        type=str,
    )

    args = parser.parse_args()
    main(args=args)
