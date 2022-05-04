import argparse

import numpy as np
from torchvision import datasets, transforms

import trainers
from evaluators.base import PredictionBasedEvaluator
from metrics.accuracy import Accuracy
from models.cifar10_models import VGG
from target_maps.analytical import ALL_ONES, ALL_ZEROS
from target_maps.comical import GOBLIN, CARROT, SWORD, BOW_ARROW, XMASS_TREE
from trainers.base import TrainerConfig
from trainers.maptraining import MapTrainingConfig
from trainers.pretraining import Trainer
from utilities.utils import xavier_uniform_initialize, gpu_information_summary, set_seed, safely_load_state_dict

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
    n_gpu, device = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)
    cifar_data_train = datasets.CIFAR10(root=args.data_root, train=True, transform=cifar_transform_train, download=True)
    cifar_data_valid = datasets.CIFAR10(
        root=args.data_root, train=False, transform=cifar_transform_valid, download=True
    )
    model = VGG()
    if args.model_path:
        model.load_state_dict(safely_load_state_dict(args.model_path))
    else:
        model.apply(xavier_uniform_initialize)

    if not args.with_map:
        # This is simple pretraining
        training_config = TrainerConfig(
            prediction_evaluator=PredictionBasedEvaluator(metrics=[Accuracy()]),
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
        # This means we have a map target
        if args.experiment_name == "GoblinCKA":
            target_cka = GOBLIN
        elif args.experiment_name == "CarrotCKA":
            target_cka = CARROT
        elif args.experiment_name == "SwordCKA":
            target_cka = SWORD
        elif args.experiment_name == "BowArrowCKA":
            target_cka = BOW_ARROW
        elif args.experiment_name == "XMassTreeCKA":
            target_cka = XMASS_TREE
        elif args.experiment_name == "AllOnesCKA":
            target_cka = ALL_ONES
        elif args.experiment_name == "AllZerosCKA":
            target_cka = ALL_ZEROS
        elif args.experiment_name == "SingleOneCKA":
            target_cka = np.load(args.cka_path)
            target_cka[7, 0] = 1
            target_cka[0, 7] = 1
        else:
            raise Exception("Experiment name provided is not supported.", args.experiment_name)

        # Now we need to see whether we want to train with hard or soft labels (CE vs. Distillation)
        teacher_model = None
        if not args.with_hard_labels:
            teacher_model = VGG()
            teacher_model.load_state_dict(safely_load_state_dict(args.model_path))
            teacher_model.to(device)
            teacher_model.eval()

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
            nb_warmup_steps=0,
            learning_rate=1e-3,
            experiment_name=args.experiment_name,
            nb_classes=10,
            max_grad_norm=1.0,
            progress_history=1,
            cka_alpha=500.0,
            cka_difference_function="LogCosh",
            target_cka=target_cka,
            distillation_temp=0.1,
            teacher_model=teacher_model,
            hard_labels=True if args.with_hard_labels else False,
            upper_bound_acc=85.9,
            acc_tolerance=1.0,
            reduction_factor=0.8,
        )
        if args.with_hard_labels:
            trainer = trainers.maptraining.CEMapTrainer(
                model=model, train_dataset=cifar_data_train, valid_dataset=cifar_data_valid, config=training_config
            )
        else:
            trainer = trainers.maptraining.DistillMapTrainer(
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
    parser.add_argument(
        "--with_hard_labels",
        help="This indicates whether we want to manipulate a model with hard label or soft label."
        "Hard label trains with CE vs. soft label uses distillation to train.",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    main(args=args)
