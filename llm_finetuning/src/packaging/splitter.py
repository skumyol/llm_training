import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def _episode_hash(episode_id: str) -> int:
    return int(hashlib.md5(episode_id.encode()).hexdigest(), 16)


class Splitter:
    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_by_episode(
        self,
        packaged_dir: str,
        splits_dir: str,
    ) -> dict:
        packaged_dir = Path(packaged_dir)
        splits_dir = Path(splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)

        episode_to_turns: dict[str, list[dict]] = defaultdict(list)
        episode_scenario: dict[str, str] = {}

        for artifact in ["full_trace.jsonl", "head_supervision.jsonl", "sft.jsonl"]:
            path = packaged_dir / artifact
            if not path.exists():
                continue
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    episode_id = record.get("episode_id", "unknown")
                    key = f"{artifact}|{episode_id}"
                    episode_to_turns[key].append(record)
                    if artifact == "full_trace.jsonl":
                        episode_scenario[episode_id] = record.get("scenario_type", "unknown")

        trace_episodes = sorted({k.split("|")[1] for k in episode_to_turns if k.startswith("full_trace")})
        stratified = self._stratify_by_scenario(trace_episodes, episode_scenario)
        assignment = self._assign_splits(stratified)

        artifact_names = ["full_trace", "head_supervision", "sft"]
        split_names = ["train", "val", "test"]
        split_file_map: dict[str, dict[str, Path]] = {a: {} for a in artifact_names}

        for artifact_base in artifact_names:
            for split in split_names:
                suffix_map = {
                    "full_trace": "trace",
                    "head_supervision": "heads",
                    "sft": "sft",
                }
                fname = splits_dir / f"{split}_{suffix_map[artifact_base]}.jsonl"
                split_file_map[artifact_base][split] = fname

        file_handles: dict = {}
        for artifact_base in artifact_names:
            for split in split_names:
                file_handles[f"{artifact_base}|{split}"] = open(
                    split_file_map[artifact_base][split], "w"
                )

        counts: dict = {s: 0 for s in split_names}
        for artifact_base in artifact_names:
            artifact_file = artifact_base + ".jsonl"
            key_prefix = artifact_file
            for key, turns in episode_to_turns.items():
                if not key.startswith(key_prefix + "|"):
                    continue
                episode_id = key.split("|", 1)[1]
                split = assignment.get(episode_id, "train")
                fh = file_handles[f"{artifact_base}|{split}"]
                for t in turns:
                    fh.write(json.dumps(t) + "\n")
                if artifact_base == "full_trace":
                    counts[split] += len(turns)

        for fh in file_handles.values():
            fh.close()

        manifest = {
            "split_counts": counts,
            "n_train_episodes": sum(1 for v in assignment.values() if v == "train"),
            "n_val_episodes":   sum(1 for v in assignment.values() if v == "val"),
            "n_test_episodes":  sum(1 for v in assignment.values() if v == "test"),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
        }
        with open(splits_dir / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Split complete: {manifest}")
        return manifest

    def _stratify_by_scenario(
        self,
        episodes: list[str],
        episode_scenario: dict[str, str],
    ) -> dict[str, list[str]]:
        strata: dict[str, list[str]] = defaultdict(list)
        for ep in episodes:
            st = episode_scenario.get(ep, "unknown")
            strata[st].append(ep)
        return dict(strata)

    def _assign_splits(self, stratified: dict[str, list[str]]) -> dict[str, str]:
        assignment: dict[str, str] = {}
        rng = random.Random(self.seed)
        for scenario_type, episodes in stratified.items():
            shuffled = list(episodes)
            rng.shuffle(shuffled)
            n = len(shuffled)
            if n == 1:
                assignment[shuffled[0]] = "train"
                continue
            if n == 2:
                assignment[shuffled[0]] = "train"
                assignment[shuffled[1]] = "val"
                continue
            n_train = max(1, int(n * self.train_ratio))
            n_val   = max(1, int(n * self.val_ratio))
            
            # If we don't have enough for all sets, prioritize Train > Val > Test
            if n_train + n_val > n:
                # If we can't fit both, shrink train if strictly necessary, or just shrink val?
                # Actually, standard behavior for small N:
                # n=3 (0.8, 0.1, 0.1) -> 2 train, 1 val, 0 test is better than 2 train, 0 val, 1 test.
                if n_train + n_val > n:
                    # Try to keep val=1 if possible
                    if n_val == 1 and n > 1:
                         n_train = n - 1
                    else:
                         n_val = n - n_train
            
            for ep in shuffled[:n_train]:
                assignment[ep] = "train"
            for ep in shuffled[n_train:n_train + n_val]:
                assignment[ep] = "val"
            for ep in shuffled[n_train + n_val:]:
                assignment[ep] = "test"
        return assignment
