"""NumPy-only CNN gaze estimator with a baseline-initialized linear skip branch."""

import argparse
import os
import pickle
import sys

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMMON_ROOT = os.path.join(ROOT, "deep_learning_from_scratch")
for path in (HERE, COMMON_ROOT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import common.layers as common_layers
import common.optimizer as common_optimizer
import common.util as common_util
from baseline_linear import PERSONS, Stats, angular_error, fit_ridge, load_person


def set_backend(device):
    """Switch the scratch layers to CuPy when GPU execution is requested."""
    gpu_ready = False
    if CUPY_AVAILABLE:
        try:
            gpu_ready = cp.cuda.runtime.getDeviceCount() > 0
        except cp.cuda.runtime.CUDARuntimeError:
            gpu_ready = False
    use_gpu = device == "gpu" or (device == "auto" and gpu_ready)
    if use_gpu and not CUPY_AVAILABLE:
        raise RuntimeError(
            "--device gpu requires CuPy. Install a CUDA-matched cupy package."
        )
    if device == "gpu" and not gpu_ready:
        raise RuntimeError("--device gpu requested, but no CUDA GPU is available.")
    if use_gpu:
        common_layers.np = cp
        common_optimizer.np = cp
        common_util.np = cp
        return cp, "gpu"
    return np, "cpu"


class GazeCNN:
    """Frozen baseline plus a small, regularized CNN residual."""

    def __init__(
        self, ridge_W, ridge_b, xp, seed=0, residual_scale=0.1, residual_l2=1e-3
    ):
        self.xp = xp
        self.residual_scale = residual_scale
        self.residual_l2 = residual_l2
        rng = xp.random.RandomState(seed) if xp is np else xp.random.RandomState(seed)
        randn = rng.randn
        Affine, Convolution, Pooling, Relu = (
            common_layers.Affine,
            common_layers.Convolution,
            common_layers.Pooling,
            common_layers.Relu,
        )
        self.skip_W = xp.asarray(ridge_W, dtype=xp.float32)
        self.skip_b = xp.asarray(ridge_b, dtype=xp.float32)
        self.conv = Convolution(
            (randn(8, 1, 5, 5) * 0.02).astype(xp.float32),
            xp.zeros(8, dtype=xp.float32),
            pad=2,
        )
        self.relu1 = Relu()
        self.pool = Pooling(2, 2, stride=2)
        self.conv2 = Convolution(
            (randn(16, 8, 3, 3) * 0.02).astype(xp.float32),
            xp.zeros(16, dtype=xp.float32),
            pad=1,
        )
        self.relu2 = Relu()
        self.fc = Affine(
            (randn(16 * 18 * 30, 2) * 0.01).astype(xp.float32),
            xp.zeros(2, dtype=xp.float32),
        )
        # At initialization the model exactly reproduces the baseline prediction.
        self.fc.W.fill(0.0)
        # Keep the already strong Ridge baseline fixed. Only the residual CNN learns.
        self.params = {
            "conv_W": self.conv.W,
            "conv_b": self.conv.b,
            "conv2_W": self.conv2.W,
            "conv2_b": self.conv2.b,
            "fc_W": self.fc.W,
            "fc_b": self.fc.b,
        }

    def predict(self, eye, head):
        x = self.conv.forward(eye)
        x = self.relu1.forward(x)
        x = self.pool.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        residual = self.fc.forward(x)
        self.direct = self.xp.concatenate((eye.reshape(len(eye), -1), head), axis=1)
        self.residual = residual
        return self.direct @ self.skip_W + self.skip_b + self.residual_scale * residual

    def loss_and_grads(self, eye, head, target):
        pred = self.predict(eye, head)
        target = self.xp.asarray(target, dtype=self.xp.float32)
        # Cosine loss on the 3-D unit gaze vectors, matching the evaluation metric.
        yaw, pitch = pred[:, 0], pred[:, 1]
        cy, sy, cpitch, spitch = (
            self.xp.cos(yaw),
            self.xp.sin(yaw),
            self.xp.cos(pitch),
            self.xp.sin(pitch),
        )
        vec = self.xp.stack((-cpitch * sy, -spitch, -cpitch * cy), axis=1)
        ty, tp = target[:, 0], target[:, 1]
        target_vec = self.xp.stack(
            (
                -self.xp.cos(tp) * self.xp.sin(ty),
                -self.xp.sin(tp),
                -self.xp.cos(tp) * self.xp.cos(ty),
            ),
            axis=1,
        )
        cosine = self.xp.sum(vec * target_vec, axis=1)
        loss = self.xp.mean(1.0 - cosine) + self.residual_l2 * self.xp.mean(
            self.residual**2
        )
        # d(unit_vector)/d(yaw), d(unit_vector)/d(pitch)
        d_yaw = self.xp.stack(
            (-cpitch * cy, self.xp.zeros_like(yaw), cpitch * sy), axis=1
        )
        d_pitch = self.xp.stack((spitch * sy, -cpitch, spitch * cy), axis=1)
        d_pred = -self.xp.stack(
            (
                self.xp.sum(d_yaw * target_vec, axis=1),
                self.xp.sum(d_pitch * target_vec, axis=1),
            ),
            axis=1,
        ) / len(target)
        dout = (
            self.residual_scale * d_pred
            + 2.0 * self.residual_l2 * self.residual / len(target)
        )
        grads = {}
        dx = self.fc.backward(dout)
        dx = self.relu2.backward(dx)
        dx = self.conv2.backward(dx)
        dx = self.pool.backward(dx)
        dx = self.relu1.backward(dx)
        self.conv.backward(dx)
        grads.update(
            {
                "conv_W": self.conv.dW,
                "conv_b": self.conv.db,
                "conv2_W": self.conv2.dW,
                "conv2_b": self.conv2.db,
                "fc_W": self.fc.dW,
                "fc_b": self.fc.db,
            }
        )
        return float(loss), grads

    def state_dict(self):
        return {
            key: (cp.asnumpy(value) if self.xp is cp else value.copy())
            for key, value in self.params.items()
        }


def predict_in_batches(model, X, batch_size, xp):
    """Run inference in small batches to avoid allocating a huge im2col buffer."""
    predictions = []
    for start in range(0, len(X), batch_size):
        batch = X[start : start + batch_size]
        pred = model.predict(
            xp.asarray(batch[:, :2160].reshape(-1, 1, 36, 60), dtype=xp.float32),
            xp.asarray(batch[:, 2160:], dtype=xp.float32),
        )
        predictions.append(cp.asnumpy(pred) if xp is cp else pred)
    return np.concatenate(predictions, axis=0)


def run_fold(root, test_person, epochs, batch_size, lr, seed, xp):
    train_people = [p for p in PERSONS if p != test_person]
    # baseline_linear.load_person uses ThreadPoolExecutor and requires workers > 0.
    train_sets = [
        load_person(root, p, use_head=True, num_workers=8) for p in train_people
    ]
    test_X, test_y = load_person(root, test_person, use_head=True, num_workers=8)
    X = np.concatenate([item[0] for item in train_sets])
    y = np.concatenate([item[1] for item in train_sets])
    W, b = fit_ridge(Stats(X, y), alpha=1000.0)
    model = GazeCNN(W, b, xp=xp, seed=seed)
    optimizer = common_optimizer.Adam(lr=lr)
    rng = np.random.default_rng(seed)
    initial_pred = predict_in_batches(model, test_X, batch_size, xp)
    initial_error = float(angular_error(initial_pred, test_y).mean())
    print(f"  {test_person} initial (baseline): {initial_error:.2f} degrees")
    for epoch in range(epochs):
        order, losses = rng.permutation(len(y)), []
        for start in range(0, len(y), batch_size):
            idx = order[start : start + batch_size]
            eye = xp.asarray(X[idx, :2160].reshape(-1, 1, 36, 60), dtype=xp.float32)
            head = xp.asarray(X[idx, 2160:], dtype=xp.float32)
            loss, grads = model.loss_and_grads(eye, head, y[idx])
            optimizer.update(model.params, grads)
            losses.append(loss)
        print(f"  {test_person} epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.6f}")
    pred = predict_in_batches(model, test_X, batch_size, xp)
    return model, float(angular_error(pred, test_y).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join(HERE, "data/EyeBased/MPIIGaze"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default=os.path.join(HERE, "gaze_cnn_models.pkl"))
    parser.add_argument("--fold", choices=PERSONS)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args()
    xp, backend = set_backend(args.device)
    print(f"device: {backend}")
    folds = [args.fold] if args.fold else PERSONS
    models, errors = {}, {}
    for i, person in enumerate(folds):
        model, errors[person] = run_fold(
            args.root,
            person,
            args.epochs,
            args.batch_size,
            args.lr,
            seed=100 + i,
            xp=xp,
        )
        models[person] = model.state_dict()
        print(f"  test {person}: {errors[person]:.2f} degrees")
        # Save each fold separately: *_01.pkl, *_02.pkl, ...
        output_root, output_ext = os.path.splitext(args.output)
        fold_output = f"{output_root}_{i + 1:02d}{output_ext or '.pkl'}"
        with open(fold_output, "wb") as handle:
            pickle.dump(
                {
                    "models": {person: models[person]},
                    "errors": {person: errors[person]},
                    "test_person": person,
                    "config": vars(args),
                    "model_type": "GazeCNN(frozen_ridge_plus_residual)",
                    "backend": backend,
                },
                handle,
            )
        print(f"  saved model: {fold_output}")
    print("\n| 테스트 대상 | " + " | ".join(folds) + " | 평균 |")
    print("|-------------|" + "-----|" * (len(folds) + 1))
    print(
        "| 각도 오차 (°) | "
        + " | ".join(f"{errors[p]:.2f}" for p in folds)
        + f" | {np.mean(list(errors.values())):.2f} |"
    )
    with open(args.output, "wb") as handle:
        pickle.dump(
            {
                "models": models,
                "errors": errors,
                "config": vars(args),
                "model_type": "GazeCNN(frozen_ridge_plus_residual)",
                "backend": backend,
            },
            handle,
        )
    print(f"mean angular error: {np.mean(list(errors.values())):.2f} degrees")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
