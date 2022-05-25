import argparse
import os

import numpy as np
from torchvision import datasets, transforms
from evaluators.base import BatchRepresentationBasedEvaluator, RepresentationBasedEvaluator
from metrics.cka import BatchCKA, CKA
from models.cifar_10_models.resnet import ResNet34
from models.cifar_10_models.vgg import VGG
from utilities.utils import xavier_uniform_initialize, gpu_information_summary, set_seed, safely_load_state_dict
from datasets.pcam import PCAM

normalize_cifar = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

cifar_transform_valid = transforms.Compose([transforms.ToTensor(), normalize_cifar])
patch_camelyon_transform_valid = transforms.Compose([transforms.CenterCrop(32), transforms.ToTensor()])


def main(args):
    n_gpu, device = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)

    if args.dataset.lower() == "cifar10":
        under_study_data = datasets.CIFAR10(
            root=args.data_root, train=False, transform=cifar_transform_valid, download=True
        )
    elif args.dataset.lower() == "patchcamelyon":
        under_study_data = PCAM(root=args.data_root, split="test", transform=cifar_transform_valid, download=True)
    else:
        error_message = f"Unsupported --dataset: {args.dataset}"
        raise ValueError(error_message)

    if args.model_type.lower() == "vgg":
        model = VGG(width=args.network_width)
    elif args.model_type.lower() == "resnet":
        model = ResNet34(width=args.network_width)
    else:
        error_message = f"Unsupported --model_type: {args.model_type}"
        raise Exception(error_message)

    if args.model_path:
        model.load_state_dict(safely_load_state_dict(args.model_path))
    else:
        model.apply(xavier_uniform_initialize)

    second_model = None
    if args.second_model_path is not None:
        if args.second_model_type.lower() == "vgg":
            second_model = VGG(width=args.second_network_width)
        elif args.second_model_type.lower() == "resnet":
            second_model = ResNet34(width=args.second_network_width)
        else:
            error_message = f"Unsupported --second_model_type: {args.second_model_type}"
            raise Exception(error_message)

        second_model = VGG(width=args.second_network_width)
        second_model.load_state_dict(safely_load_state_dict(args.second_model_path))

    if args.rbf_sigma == -1:
        representation_evaluator = BatchRepresentationBasedEvaluator(
            metrics=[BatchCKA()], batch_size=args.batch_size, num_workers=args.num_workers
        )

        cka_results = representation_evaluator.evaluate(model_1=model, dataset=under_study_data, model_2=second_model)[
            "BatchCKA"
        ]
    else:
        representation_evaluator = RepresentationBasedEvaluator(
            metrics=[CKA(rbf_sigma=args.rbf_sigma)],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        representation_evaluator.record_representations_set_1(model=model, dataset=under_study_data)
        if second_model is not None:
            representation_evaluator.record_representations_set_1(model=second_model, dataset=under_study_data)
        cka_results = representation_evaluator.compute_metrics()["CKA"]
    with np.printoptions(precision=3, suppress=True):
        print(cka_results)

    np.save(os.path.join(args.output_path, f"CKA_{args.experiment_name}.npy"), cka_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        help="Path to the root data.",
        type=str,
    )

    parser.add_argument(
        "--batch_size",
        help="Evaluation batch size.",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers to fetch data.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--model_path",
        help="Path of the model the to evaluate.",
        type=str,
    )
    parser.add_argument("--second_model_path", help="Path of the second model to evaluate.", type=str, default=None)
    parser.add_argument("--experiment_name", help="Name for the experiment.", type=str, default="cka_evaluation")

    parser.add_argument(
        "--network_width",
        help="x times the width of the base model.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--second_network_width",
        help="x times the width of the base model.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the results.",
        type=str,
    )

    parser.add_argument(
        "--seed_value",
        help="Random seed value for this experiment.",
        type=int,
        default=3407,
    )

    parser.add_argument(
        "--rbf_sigma",
        help="If the RBF sigma is == -1 then the linear CKA is computed, else the kernel CKA is computed and the sigma"
        "will indicate the multiplier to the median distance.",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--model_type",
        help="Indicate the model architecture type from the available choices.",
        type=str,
        choices=["VGG", "ResNet"],
        default="VGG",
    )

    parser.add_argument(
        "--second_model_type",
        help="Indicate the model architecture type from the available choices.",
        type=str,
        choices=["VGG", "ResNet"],
        default="VGG",
    )

    parser.add_argument(
        "--dataset",
        help="Indicate the name of the dataset of interest.",
        type=str,
        choices=["CIFAR10", "PatchCamelyon"],
        default="CIFAR10",
    )

    args = parser.parse_args()
    main(args=args)
