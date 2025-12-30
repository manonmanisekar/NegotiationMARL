# ctde_rms_project/env/negotiation.py
import random
import numpy as np

class MultiLevelNegotiation:
    """
    Multi-level negotiation for CTDE RMS:
      - level1: machine-local negotiation (machine bids based on speed, setup, reputation)
      - level2: cell-level negotiation (machines grouped into cells; balance load across machines in a cell)
      - level3: global negotiation (move jobs across cells, trigger reconfiguration)
    This is a modular, extensible implementation:
      - each level returns a candidate machine index (or None)
      - the environment coordinates the final assignment
    """

    def __init__(self, env, cell_size=2, rng_seed=None,
                 level1_weight=0.6, level2_weight=0.3, level3_weight=0.1):
        self.env = env
        self.cell_size = max(1, int(cell_size))
        self.rng = random.Random(rng_seed)
        self.level1_w = level1_weight
        self.level2_w = level2_weight
        self.level3_w = level3_weight

        # Build cell partitioning (simple contiguous partition)
        nm = len(self.env.machines)
        self.cells = []
        for i in range(0, nm, self.cell_size):
            self.cells.append(list(range(i, min(i + self.cell_size, nm))))

    # -------------------------
    # Level 1: machine bidding
    # -------------------------
    def level1_machine_bids(self, job, eligible_indices):
        """
        Each eligible machine returns a 'bid' score.
        Higher score => more desirable. Score factors:
          - speed (positive)
          - reputation (positive)
          - current utilization (negative)
          - setup (negative)
          - closeness to deadline (urgency term)
        Returns: list of tuples (machine_idx, score, setup, expected_ptime)
        """
        bids = []
        current_time = self.env.time
        for mid in eligible_indices:
            m = self.env.machines[mid]
            setup = self.env.setup_time(m, job)
            expected_ptime = job.processing_time / m.speed
            urgency = max(0.0, (current_time + setup + expected_ptime) - job.deadline)
            score = (
                m.speed * 1.5 + m.reputation * 2.0
                - 0.8 * (m.utilization / (1 + sum([mm.utilization for mm in self.env.machines])))
                - 1.2 * setup
                - 1.7 * urgency
            )
            # Add slight randomness to avoid ties
            score += self.rng.uniform(-0.05, 0.05)
            bids.append((mid, float(score), int(setup), float(expected_ptime)))
        return bids

    # -------------------------
    # Level 2: cell negotiation
    # -------------------------
    def level2_cell_select(self, job, bids):
        """
        Aggregate machine bids at cell level.
        Strategy:
          - compute cell-level score as mean of top-k machine scores in the cell
          - prefer cells with lower average utilization
        Returns: candidate machine index (within original bids) from the best cell
        """
        # Map machine->bid
        bid_map = {b[0]: b for b in bids}
        cell_scores = []
        for cell in self.cells:
            cell_bids = [bid_map[mid][1] for mid in cell if mid in bid_map]
            if not cell_bids:
                continue
            # prefer cells where machines have good scores and low utilization
            cell_util = np.mean([self.env.machines[mid].utilization for mid in cell])
            cell_score = np.mean(sorted(cell_bids, reverse=True)[:max(1, len(cell_bids)//2)]) - 0.5 * (cell_util)
            cell_scores.append((cell, float(cell_score)))
        if not cell_scores:
            return None
        # choose best cell
        cell_scores.sort(key=lambda x: x[1], reverse=True)
        best_cell = cell_scores[0][0]
        # within cell, pick machine with highest bid
        candidates = [bid_map[mid] for mid in best_cell if mid in bid_map]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = candidates[0][0]
        return int(chosen)

    # -------------------------
    # Level 3: global negotiation
    # -------------------------
    def level3_global_decision(self, job, current_best_mid):
        """
        Global coordination that can:
         - consider moving job to another cell if global imbalance exists
         - trigger reconfiguration (heavy penalty) only if it yields clear benefit
        Strategy (simple heuristic):
          - compute global utilization std; if too imbalanced, allow migration
          - else, trust lower-level decisions
        Returns: machine idx (may be same as current_best_mid)
        """
        nm = len(self.env.machines)
        utilizations = np.array([m.utilization for m in self.env.machines])
        if len(utilizations) == 0:
            return current_best_mid
        util_std = float(np.std(utilizations))
        # if imbalance high, try to find a machine in low-util cell that can accept job
        if util_std > 0.25:
            # find cells with low mean utilization
            cell_util = [(cell, np.mean([self.env.machines[mid].utilization for mid in cell])) for cell in self.cells]
            cell_util.sort(key=lambda x: x[1])
            # attempt to place in lowest-util cell if capability exists and not too costly in setup
            for cell, u in cell_util[:2]:
                for mid in cell:
                    if job.process_type in self.env.machines[mid].capabilities:
                        setup = self.env.setup_time(self.env.machines[mid], job)
                        expected_ptime = job.processing_time / self.env.machines[mid].speed
                        # allow if setup not excessive relative to expected_ptime
                        if setup <= max(3, 0.5 * expected_ptime):
                            return int(mid)
        # otherwise keep current
        return current_best_mid

    # -------------------------
    # Negotiation orchestrator
    # -------------------------
    def negotiate(self, job):
        """
        High-level method invoked by environment per job.
        Returns:
          - chosen_machine_idx (int) or None if no eligible machine
          - diagnostics dict with bids and decisions per level
        """
        eligible = [i for i,m in enumerate(self.env.machines) if job.process_type in m.capabilities]
        diag = {'eligible': eligible, 'level1': [], 'level2_choice': None, 'level3_choice': None}
        if not eligible:
            return None, diag

        # Level 1: machine bidding
        bids = self.level1_machine_bids(job, eligible)
        diag['level1'] = bids

        # select top machine from level1 by weighted score
        bids_sorted = sorted(bids, key=lambda x: x[1], reverse=True)
        l1_choice = bids_sorted[0][0] if bids_sorted else None
        diag['level1_choice'] = l1_choice

        # Level 2: cell selection
        l2_choice = self.level2_cell_select(job, bids)
        diag['level2_choice'] = l2_choice

        # Combine level1 and level2 via simple weighted vote
        # if both exist, compute combined preference score for candidates
        candidates = set([i for i in [l1_choice, l2_choice] if i is not None])
        if not candidates:
            combined_choice = l1_choice or l2_choice or bids_sorted[0][0]
        else:
            # choose candidate with highest combined weight:
            combined_scores = {}
            for c in candidates:
                s1 = next((b[1] for b in bids if b[0]==c), 0.0)
                s2 = 0.0
                # cell score heuristic
                for cell,score in [(cell, sum([next((b[1] for b in bids if b[0]==mid),0.0) for mid in cell]) / max(1,len(cell))) for cell in self.cells]:
                    if c in cell:
                        s2 = score
                        break
                combined_scores[c] = self.level1_w * s1 + self.level2_w * s2
            # pick best combined
            combined_choice = max(combined_scores.items(), key=lambda x: x[1])[0]

        # Level 3: global decision may override
        l3_choice = self.level3_global_decision(job, combined_choice)
        diag['level3_choice'] = l3_choice

        # final chosen machine
        chosen = int(l3_choice)
        diag['chosen'] = chosen
        return chosen, diag
