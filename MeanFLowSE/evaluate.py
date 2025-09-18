# evaluate.py
import glob
from os.path import join
from argparse import ArgumentParser

import torch
from torchaudio import load
from tqdm import tqdm
from soundfile import write

from flowmse.model import VFModel
from flowmse.data_module import SpecsDataModule  
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver
from utils import ensure_dir


def _auto_pick_solver(model, user_choice):
    """
    根据模型开关与用户显式传入的采样器名称来确定最终采样器。
    - 若用户明确指定，则尊重用户；
    - 否则：use_mfse==True -> 'euler_mf'；反之 'euler'。
    """
    if user_choice is not None:
        return user_choice
    use_mf = getattr(model, "use_mfse", False)
    return "euler_mf" if use_mf else "euler"


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing the test data (must have test/{clean,noisy}).")
    parser.add_argument("--folder_destination", type=str, required=True,
                        help="Destination path of inference results. Absolute path is required.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt).")


    parser.add_argument("--odesolver_type", type=str, choices=("white",), default="white",
                        help="Sampler family. We use 'white' (Euler family).")

    parser.add_argument("--odesolver", type=str, choices=("euler", "euler_mf"), default=None,
                        help="Numerical integrator: 'euler' (FlowSE) or 'euler_mf' (MeanFlow displacement). "
                             "Default: auto pick based on checkpoint config.")
    parser.add_argument("--reverse_starting_point", type=float, default=1.0,
                        help="Starting point t_N in the reverse ODE (default: 1.0).")
    parser.add_argument("--last_eval_point", type=float, default=0.03,
                        help="Terminal time t_eps for numerical stability (default: 0.03).")

    parser.add_argument("--N", type=int, default=5, help="Number of time steps (multi-step).")
    parser.add_argument("--one_step", action="store_true",
                        help="Use 1-NFE displacement (MeanFlow). If set, N is ignored and N=1.")

    parser.add_argument("--N_mid", type=int, default=0, help="It is not related to FlowSE")

    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")
    checkpoint_file = args.ckpt

    sr = 16000
    odesolver_type = args.odesolver_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VFModel.load_from_checkpoint(
        checkpoint_file,
        data_module_cls=SpecsDataModule,
        map_location=device
    )
    model = model.eval(no_ema=False).to(device)

    noisy_files = sorted(glob.glob(f"{noisy_dir}/*.wav"))

    target_dir = f"{args.folder_destination}/"
    ensure_dir(target_dir + "files/")

    solver_name = _auto_pick_solver(model, args.odesolver)

    N_eff = 1 if args.one_step else int(args.N)
    reverse_starting_point = float(args.reverse_starting_point)
    reverse_end_point = float(args.last_eval_point)

    model.T_rev = reverse_starting_point
    if hasattr(model, "ode"):
        model.ode.T_rev = reverse_starting_point

    if args.one_step and solver_name == "euler_mf" and reverse_end_point > 0:
        print(f"[Warn] 1‑NFE + euler_mf，当前 last_eval_point={reverse_end_point}，"
              f"若需严格一步从 1→0，可把 --last_eval_point 设为 0.0。")

    for _, noisy_file in tqdm(list(enumerate(noisy_files)), total=len(noisy_files)):
        filename = noisy_file.split('/')[-1]


        x, _ = load(join(clean_dir, filename))  
        y, _ = load(noisy_file)

        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(device))), 0)
        Y = pad_spec(Y)

        if odesolver_type == "white":
            sampler = get_white_box_solver(
                solver_name, model.ode, model, Y=Y, Y_prior=Y,
                T_rev=reverse_starting_point, t_eps=reverse_end_point, N=N_eff
            )
        else:
            raise ValueError(f"{odesolver_type} is not a valid sampler type!")

        with torch.no_grad():
            sample, _ = sampler()
            sample = sample.squeeze()

        x_hat = model.to_audio(sample, T_orig)
        x_hat = x_hat * norm_factor

        write(target_dir + "files/" + filename, x_hat.squeeze().cpu().numpy(), sr)
