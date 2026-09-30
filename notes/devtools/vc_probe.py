"""Minimal probe: does pl.split(UP_DOWN) survive a VECTOR -> CUBE boundary?

scaled_dot_kkt splits fine on hardware, and its only boundary is cube -> vector
(matmul result consumed by vector ops, stored to GM). wy_fast additionally has a
vector -> cube boundary: col_expand_mul feeds a matmul. That is the one
structural difference, and aic_gather -- recombining the two AIV lanes' halves
into one cube operand -- is the machinery only wy_fast exercises.

Usage: -m cv  (cube->vector only, expected OK)
       -m vc  (vector->cube, the suspect)
"""
import pypto.language as pl

M = 128
N = 128


@pl.jit
def probe_vc(a: pl.Tensor[[M, N], pl.FP16], b: pl.Tensor[[N, N], pl.FP16],
             out: pl.Out[pl.Tensor[[M, N], pl.FP16]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_vc",
               optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        scaled = pl.mul(a[:, :], 2.0)          # VECTOR
        out[:, :] = pl.matmul(scaled, b[:, :])  # CUBE consumes it -> V->C boundary
    return out


@pl.jit
def probe_cv(a: pl.Tensor[[M, N], pl.FP16], b: pl.Tensor[[N, N], pl.FP16],
             out: pl.Out[pl.Tensor[[M, N], pl.FP16]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_cv",
               optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        prod = pl.matmul(a[:, :], b[:, :])      # CUBE
        out[:, :] = pl.mul(prod, 2.0)           # VECTOR consumes it -> C->V boundary
    return out


@pl.jit
def probe_vc_nosplit(a: pl.Tensor[[M, N], pl.FP16], b: pl.Tensor[[N, N], pl.FP16],
                     out: pl.Out[pl.Tensor[[M, N], pl.FP16]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_vc_nosplit"):
        scaled = pl.mul(a[:, :], 2.0)
        out[:, :] = pl.matmul(scaled, b[:, :])
    return out


if __name__ == "__main__":
    import argparse
    import torch
    from golden import TensorSpec, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="vc", choices=["vc", "cv", "vc_nosplit"])
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    fn = {"vc": probe_vc, "cv": probe_cv, "vc_nosplit": probe_vc_nosplit}[args.mode]

    def golden(tensors):
        a = tensors["a"].float()
        b = tensors["b"].float()
        if args.mode == "cv":
            tensors["out"][:] = ((a @ b) * 2.0).half()
        else:
            tensors["out"][:] = ((a * 2.0) @ b).half()

    result = run_jit(
        fn=fn,
        specs=[TensorSpec("a", [M, N], torch.float16, init_value=torch.randn),
               TensorSpec("b", [N, N], torch.float16, init_value=torch.randn),
               TensorSpec("out", [M, N], torch.float16, is_output=True)],
        golden_fn=golden,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2, atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
