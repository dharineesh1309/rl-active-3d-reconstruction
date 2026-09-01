import os
import glob
import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, state: dict, tag: str):
        path = os.path.join(self.dir, f"ckpt_{tag}.pt")

        # Never clobber an existing 'final'. It is usually the only artifact of
        # a completed run, and a short run finishing normally would otherwise
        # overwrite tens of thousands of episodes with a handful -- which is
        # exactly what a 3-episode smoke test did here once.
        if tag == "final" and os.path.exists(path):
            existing = self._episodes_in(path)
            incoming = state.get("total_episodes", 0)
            if existing > incoming:
                alt = os.path.join(self.dir, f"ckpt_final_ep{incoming}.pt")
                print(f"  [checkpoint] {os.path.basename(path)} holds {existing} "
                      f"episodes, more than this run's {incoming}; "
                      f"saving to {os.path.basename(alt)} instead.")
                torch.save(state, alt)
                return

        torch.save(state, path)

    @staticmethod
    def _episodes_in(path: str) -> int:
        try:
            return torch.load(path, map_location="cpu",
                              weights_only=False).get("total_episodes", 0)
        except Exception:
            return 0

    def load_latest(self) -> dict | None:
        """
        The checkpoint a `--resume` should continue from.

        `ckpt_resume.pt` wins outright when present. It is written specifically
        to continue an interrupted run, so it belongs to *this* run — whereas
        ranking purely by episode count lets an unrelated older checkpoint with
        more episodes shadow it. That matters wherever sessions are capped and
        every restart is a resume: a stale `ckpt_final.pt` from a different
        experiment would otherwise be picked every time and fail to load.

        Otherwise: the highest episode count, with the per-file episode number
        read from the tag where possible so this does not deserialise every
        checkpoint in the directory just to rank them.
        """
        resume_path = os.path.join(self.dir, "ckpt_resume.pt")
        if os.path.isfile(resume_path):
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            print(f"  Resuming from ckpt_resume.pt at episode "
                  f"{state.get('total_episodes', '?')}.")
            return state

        files = glob.glob(os.path.join(self.dir, "ckpt_*.pt"))
        if not files:
            return None

        def rank(path):
            # ckpt_ep1234.pt -> 1234 without loading the file; anything else
            # (final, custom tags) falls back to reading it.
            stem = os.path.basename(path)[len("ckpt_"):-len(".pt")]
            if stem.startswith("ep") and stem[2:].isdigit():
                return int(stem[2:])
            return self._episodes_in(path)

        best_path = max(files, key=rank)
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        print(f"  Found checkpoint {os.path.basename(best_path)} at episode "
              f"{state.get('total_episodes', '?')}.")
        return state