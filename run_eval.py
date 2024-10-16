import os
from pathlib import Path
from model import PrototypeChooser
import torch
import argparse
from eval.stability import evaluate_stability
from eval.consistency import evaluate_consistency
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', default='CUB2011', type=str)
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--nb_classes', type=int, default=200)
    parser.add_argument('--test_batch_size', type=int, default=30)

    # Model
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--num_descriptive', type=int, default=10)
    parser.add_argument('--num_prototypes', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=200)

    parser.add_argument('--arch', type=str, default='resnet34')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--add_on_layers_type', type=str, default='log')
    parser.add_argument('--prototype_activation_function',
                        type=str, default='log')
    parser.add_argument('--use_thresh', action='store_true')
    parser.add_argument('--proto_depth', default=128, type=int)
    parser.add_argument('--last_layer', action='store_true')
    parser.add_argument('--inat', action='store_true')

    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    output_path = Path(f'outputs/{args.base_architecture}-{args.num_prototypes}')
    output_path.mkdir(parents=True, exist_ok=True)
    filename = 'eval_results.txt'

    img_size = args.input_size
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    base_architecture = args.base_architecture
    prototype_shape = [args.num_prototypes, 64, 1, 1]
    num_classes = 200
    prototype_activation_function = 'log'
    add_on_layers_type = 'regular'

    # Load the model
    ppnet = PrototypeChooser(
        num_prototypes=args.num_prototypes,
        num_descriptive=args.num_descriptive,
        num_classes=args.num_classes,
        use_thresh=args.use_thresh,
        arch=args.arch,
        pretrained=args.pretrained,
        add_on_layers_type=args.add_on_layers_type,
        prototype_activation_function=args.prototype_activation_function,
        proto_depth=args.proto_depth,
        use_last_layer=args.last_layer,
        inat=args.inat,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
    # ppnet.load_state_dict(checkpoint)
    ppnet.load_state_dict(checkpoint)

    ppnet.to(device)
    ppnet.eval()

    consistency_score = evaluate_consistency(ppnet, args, save_dir=output_path.as_posix())
    print('Consistency Score : {:.2f}%'.format(consistency_score))
    with open(output_path / filename, 'a') as fp:
        fp.write('Consistency Score : {:.2f}%\n'.format(consistency_score))

    stability_score = evaluate_stability(ppnet, args)
    print('Stability Score : {:.2f}%'.format(stability_score))
    with open(output_path / filename, 'a') as fp:
        fp.write('Stability Score : {:.2f}%\n'.format(stability_score))
