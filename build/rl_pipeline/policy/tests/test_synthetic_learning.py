"""
Can the pose-conditioned policy actually learn? Known-answer tasks.

After a GAE off-by-one that silently starved the view head for three full runs,
"the loss went down" is not evidence. These are tiny environments where the
optimal policy is known in advance, so failure is unambiguous and takes seconds
rather than eleven GPU-hours to detect.

    1. one-step    reward depends on a POSE, and the pose sits at a different
                   index every episode. An index-based policy cannot do better
                   than chance here; a pose-conditioned one should solve it.
                   This is the architecture's whole premise, tested directly.

    2. two-step    reward only when a SEQUENCE of two poses is acquired, and the
                   first step pays nothing. Verifies credit actually propagates
                   backwards through GAE rather than only reaching the last step.

The PPO update here is a compact reimplementation rather than the shipped
trainer, which is still wired to the old observation interface. It uses the same
clipped surrogate and the same GAE, so it tests the policy's learnability; the
shipped trainer gets integrated in Stage 2 and re-tested there.

    python policy/tests/test_synthetic_learning.py
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from policy.pose_policy import POSE_DIM, RGBPosePolicy

M = 8                      # candidate cameras per episode
FEAT = 2048


def _poses(rng, m=M):
    """Random azimuths on the circle, mild elevation/distance jitter."""
    az = rng.uniform(0, 2 * math.pi, m)
    el = rng.uniform(0.43, 0.52, m)
    d = rng.normal(0.0, 1.0, m)
    return np.stack([np.sin(az), np.cos(az), np.sin(el), np.cos(el), d],
                    axis=1).astype(np.float32), az


def _obs(poses, sel_idx, rng):
    k = max(len(sel_idx), 1)
    feats = np.zeros((1, k, FEAT), np.float32)
    sp = np.zeros((1, k, POSE_DIM), np.float32)
    sm = np.zeros((1, k), np.float32)
    for i, v in enumerate(sel_idx):
        feats[0, i] = rng.normal(0, 0.1, FEAT)
        sp[0, i] = poses[v]
        sm[0, i] = 1.0
    cm = np.ones((1, M), np.float32)
    for v in sel_idx:
        cm[0, v] = 0.0
    return {"selected_feats": torch.from_numpy(feats),
            "selected_poses": torch.from_numpy(sp),
            "selected_mask": torch.from_numpy(sm),
            "candidate_poses": torch.from_numpy(poses[None]),
            "candidate_mask": torch.from_numpy(cm)}


def _ppo(policy, rollout, opt, clip=0.2, epochs=4):
    """Clipped surrogate with a value term, on a flat batch of transitions."""
    obs, acts, old_lp, adv, ret = rollout
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(epochs):
        lp, v, ent = policy.evaluate_actions(obs, acts)
        ratio = torch.exp(lp - old_lp)
        l1, l2 = -adv * ratio, -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
        loss = torch.max(l1, l2).mean() \
            + 0.5 * torch.nn.functional.mse_loss(v, ret) \
            - 0.01 * ent.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()


def test_one_step_pose_target():
    """
    Reward 1.0 for acquiring the camera nearest azimuth 0, which lands at a
    DIFFERENT index every episode.

    An index policy can only guess: the paying slot moves. A pose policy can
    read the geometry. Solving this is the architecture's premise.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = RGBPosePolicy(n_candidates=M)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    def episode():
        poses, az = _poses(rng)
        target = int(np.argmin(np.minimum(az, 2 * math.pi - az)))
        o = _obs(poses, [], rng)
        with torch.no_grad():
            a, lp, v = policy.act(o)
        r = 1.0 if int(a) == target else 0.0
        return o, a, lp, v, r, int(a) == target

    hits = []
    for it in range(220):
        O, A, LP, V, R = [], [], [], [], []
        for _ in range(16):
            o, a, lp, v, r, hit = episode()
            O.append(o); A.append(a); LP.append(lp); V.append(v); R.append(r)
            hits.append(hit)
        obs = {k: torch.cat([o[k] for o in O]) for k in O[0]}
        acts = torch.cat(A)
        ret = torch.tensor(R, dtype=torch.float32)
        val = torch.cat(V)
        _ppo(policy, (obs, acts, torch.cat(LP), ret - val, ret), opt)

    early, late = np.mean(hits[:400]), np.mean(hits[-400:])
    chance = 1.0 / M
    print(f"   accuracy {early:.3f} -> {late:.3f}   (chance {chance:.3f})")
    assert late > 3 * chance, (
        f"pose-target accuracy {late:.3f} is not clearly above chance "
        f"{chance:.3f}: the policy is not reading candidate geometry")
    assert late > early + 0.10, f"no learning: {early:.3f} -> {late:.3f}"
    print("OK - one-step pose target learned (index alone could not do this)")


def test_delayed_reward_reaches_first_step():
    """
    Two steps; the reward depends ONLY on the FIRST choice and is paid at the END.

    This isolates the property that matters: does terminal reward propagate back
    through GAE to an earlier decision? The second action is irrelevant noise, so
    the task is exactly as hard as the one-step version EXCEPT that the signal
    arrives a step late. If accuracy at step 0 rises, credit propagates.

    An earlier draft asked for a two-action SEQUENCE instead. That pays only
    2/(8*7) = 3.6% of the time under a random policy, so a few thousand episodes
    produce too few positives to learn from -- it measured sample budget, not
    credit assignment.
    """
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    policy = RGBPosePolicy(n_candidates=M)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    gamma, lam = 0.99, 0.95

    def episode():
        poses, az = _poses(rng)
        target = int(np.argmin(np.minimum(az, 2 * math.pi - az)))
        steps, sel = [], []
        for _ in range(2):
            o = _obs(poses, sel, rng)
            with torch.no_grad():
                a, lp, v = policy.act(o)
            sel.append(int(a))
            steps.append((o, a, lp, v))
        hit = sel[0] == target            # ONLY the first choice matters
        return steps, (1.0 if hit else 0.0), hit

    hits = []
    for _ in range(220):
        O, A, LP, ADV, RET = [], [], [], [], []
        for _ in range(16):
            steps, r, hit = episode()
            hits.append(hit)
            vals = [float(s[3]) for s in steps] + [0.0]
            gae, advs = 0.0, [0.0, 0.0]
            for t in (1, 0):
                rew = r if t == 1 else 0.0          # paid only at the end
                nonterm = 0.0 if t == 1 else 1.0
                delta = rew + gamma * vals[t + 1] * nonterm - vals[t]
                gae = delta + gamma * lam * nonterm * gae
                advs[t] = gae
            for t, (o, a, lp, v) in enumerate(steps):
                O.append(o); A.append(a); LP.append(lp)
                ADV.append(advs[t]); RET.append(advs[t] + float(v))
        obs = {k: torch.cat([o[k] for o in O]) for k in O[0]}
        _ppo(policy, (obs, torch.cat(A), torch.cat(LP),
                      torch.tensor(ADV, dtype=torch.float32),
                      torch.tensor(RET, dtype=torch.float32)), opt)

    early, late = np.mean(hits[:400]), np.mean(hits[-400:])
    chance = 1.0 / M
    print(f"   step-0 accuracy {early:.3f} -> {late:.3f}   (chance {chance:.3f})")
    assert late > 2.5 * chance, (
        f"step-0 accuracy {late:.3f} vs chance {chance:.3f}: the terminal reward "
        "is not reaching the first decision -- exactly the failure the GAE "
        "off-by-one produced.")
    assert late > early + 0.10, f"no learning: {early:.3f} -> {late:.3f}"
    print("OK - delayed reward reaches the first step (credit propagates)")


if __name__ == "__main__":
    print("Running synthetic learning tests...\n")
    test_one_step_pose_target()
    test_delayed_reward_reaches_first_step()
    print("\nSynthetic learning tests passed.")
