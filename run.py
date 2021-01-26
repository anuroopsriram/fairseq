from copy import deepcopy
import submit


# lrs = [[0.000005], [0.00005], [0.0005], [0.005], [0.05]]
lrs = [[0.00005], [0.0005], [0.005]]

sweeps1 = [
    (
        f"lr{lr[0]}.noquant",
        {
            f'optimization.lr': f'"{lr}"'
        }
    )
    for lr in lrs
]

sweeps2 = [
    (
        f"lr{lr[0]}.quant.losswts0.1,10",
        {
            "optimization.lr": f'"{lr}"',
            "model.quantize": True,
            "criterion.loss_weights": '"[0.1,10]"',
        }
    )
    for lr in lrs
]

sweeps = sweeps1 + sweeps2
base_config = "siamese_wav2vec2_base_librispeech_1x100k.yaml"

parser = submit.create_parser()
base_args = parser.parse_args()
base_args.nodes = 1

for name, overrides in sweeps:
    args = deepcopy(base_args)
    args.name = name
    submit.main(args, base_config, overrides)
