import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


# 칸은 헤더 이름으로 찾는다. 3DGaze 칸이 있는 예전 라벨과 없는 현재 라벨을 모두 읽는다.
COLUMNS = ("Image", "2DGaze", "2DHead")
# 헤더가 없으면 현재 preprocess/mpii.py 형식의 위치를 쓴다.
DEFAULT_INDEX = (0, 4, 5)


def read_label(path, header=True):
    with open(path) as f:
        lines = [ln.strip().split(" ") for ln in f if ln.strip()]
    if header:
        names = lines.pop(0)
        index = [names.index(c) for c in COLUMNS]
    else:
        index = DEFAULT_INDEX
    return [[line[i] for i in index] for line in lines]


class loader:
    def __init__(self, path, root, header=True):
        paths = path if isinstance(path, list) else [path]
        self.lines = []
        for p in paths:
            self.lines.extend(read_label(p, header))

        self.root = root

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        name, gaze2d, head2d = self.lines[idx]
        eye = name

        label = np.array(gaze2d.split(","), dtype=np.float32)
        headpose = np.array(head2d.split(","), dtype=np.float32)

        img = cv2.imread(os.path.join(self.root, eye), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(os.path.join(self.root, eye))
        img = np.expand_dims(img.astype(np.float32) / 255.0, 0)

        info = {"eye": img, "head_pose": headpose, "name": name}

        return info, label


def collate(samples):
    infos, labels = zip(*samples)
    info = {
        "eye": np.stack([i["eye"] for i in infos]),
        "head_pose": np.stack([i["head_pose"] for i in infos]),
        "name": [i["name"] for i in infos],
    }
    return info, np.stack(labels)


class BatchLoader:
    """torch DataLoader 대신 쓰는 배치 반복자. (info, label)을 numpy 배열 배치로 돌려준다."""

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        seed=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        batches = [order[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()

        if self.num_workers > 0:
            # cv2.imread는 GIL을 풀기 때문에 스레드로도 이미지 읽기가 병렬로 된다.
            with ThreadPoolExecutor(self.num_workers) as pool:
                for idx in batches:
                    yield collate(list(pool.map(self.dataset.__getitem__, idx)))
        else:
            for idx in batches:
                yield collate([self.dataset[i] for i in idx])


def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
    dataset = loader(labelpath, imagepath, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    print(f"[Read Data]: Label path: {labelpath}")
    return BatchLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    import sys

    project_root = os.path.dirname(os.path.abspath(__file__))
    default_label = os.path.join(
        project_root, "data", "EyeBased", "MPIIGaze", "Label", "p00.label"
    )
    default_root = os.path.join(
        project_root, "data", "EyeBased", "MPIIGaze", "Image"
    )  # 경로 오류나서 코드 수정함

    path = sys.argv[1] if len(sys.argv) > 1 else default_label
    root = sys.argv[2] if len(sys.argv) > 2 else default_root
    d = loader(path, root)
    print(len(d))
    data, label = d[0]
    print(data["eye"].shape, data["head_pose"], label)
