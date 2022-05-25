import argparse
import os

import numpy as np
from torchvision import datasets, transforms

from evaluators.base import BatchRepresentationBasedEvaluator, RepresentationBasedEvaluator
from metrics.cka import BatchCKA, CKA
from models.cifar_10_models.vgg import VGG
from utilities.utils import xavier_uniform_initialize, gpu_information_summary, set_seed, safely_load_state_dict

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

cifar_transform_valid = transforms.Compose([transforms.ToTensor(), normalize])


def main(args):
    n_gpu, device = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)

    cifar_data_valid = datasets.CIFAR10(
        root=args.data_root, train=False, transform=cifar_transform_valid, download=True
    )
    model = VGG(width=args.network_width)
    if args.model_path:
        model.load_state_dict(safely_load_state_dict(args.model_path))
    else:
        model.apply(xavier_uniform_initialize)

    second_model = None
    if args.second_model_path is not None:
        second_model = VGG(width=args.second_network_width)
        second_model.load_state_dict(safely_load_state_dict(args.second_model_path))

    if args.rbf_sigma == -1:
        representation_evaluator = BatchRepresentationBasedEvaluator(
            metrics=[BatchCKA()], batch_size=args.batch_size, num_workers=args.num_workers
        )

        cka_results = representation_evaluator.evaluate(model_1=model, dataset=cifar_data_valid, model_2=second_model)[
            "BatchCKA"
        ]
    else:
        representation_evaluator = RepresentationBasedEvaluator(
            metrics=[CKA(rbf_sigma=args.rbf_sigma)],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        representation_evaluator.record_representations_set_1(model=model, dataset=cifar_data_valid)
        if second_model is not None:
            representation_evaluator.record_representations_set_1(model=second_model, dataset=cifar_data_valid)
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

    args = parser.parse_args()
    main(args=args)
