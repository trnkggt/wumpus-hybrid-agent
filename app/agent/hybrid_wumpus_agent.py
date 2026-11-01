from collections import deque

from app.agent.KB import KnowledgeBase
from app.agent.agent_state import AgentState
from app.agent.decision_policy import combined_risk, utility_score
from app.agent.planner import (
    direction_toward,
    plan_route,
    plan_route_to_adjacent_staging_cell,
    plan_shot,
    simulate_nav_actions,
    turns_to_face,
)
from app.domain.action import Action
from app.domain.cell import CellCoord
from app.domain.percept import Percept
from app.ml.features import extract_cell_features
from app.ml.models import PitRiskModel


class HybridWumpusAgent:
    def __init__(
        self,
        kb: KnowledgeBase,
        pit_model: PitRiskModel,
        start: CellCoord = CellCoord(0, 0),
        risk_threshold: float = 0.6,
        utility_floor: float = -8.0,
        allow_risky_entry: bool = True,
    ) -> None:
        if not 0.0 <= risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be between 0.0 and 1.0")
        self.kb = kb
        self.pit_model = pit_model
        self.plan = deque()
        self.start = start
        self.risk_threshold = risk_threshold
        self.utility_floor = utility_floor
        self.allow_risky_entry = allow_risky_entry
        self.last_frontier_scores: list[dict] = []

    def _choose_nearest_safe_unvisited(self, state: AgentState, cells: set[CellCoord]) -> CellCoord | None:
        best_cell = None
        best_route = None

        for cell in cells:
            route = plan_route(
                start_pos=state.pos,
                start_dir=state.dir,
                goals={cell},
                allowed=self.kb.safe,
                cave_geo=self.kb.cave_geo,
            )
            if not route and cell != state.pos:
                continue
            if best_route is None or len(route) < len(best_route):
                best_cell = cell
                best_route = route

        return best_cell


    def next_action(self, percept: Percept, state: AgentState) -> Action:
        self.kb.observe(state.pos, percept)
        return self._decide(percept, state)

    def _decide(self, percept: Percept, state: AgentState) -> Action:
        if self.plan:
            return self.plan.popleft()

        safe_unvisited = self.kb.safe - self.kb.visited

        if percept.glitter:
            route_home = plan_route(
                start_pos=state.pos,
                start_dir=state.dir,
                goals={self.start},
                allowed=self.kb.safe,
                cave_geo=self.kb.cave_geo,
            )
            self.plan = deque([Action.GRAB]) + route_home + deque([Action.CLIMB])
        elif safe_unvisited:
            target = self._choose_nearest_safe_unvisited(state, safe_unvisited)
            if target is not None:
                self.plan = plan_route(
                    start_pos=state.pos,
                    start_dir=state.dir,
                    goals={target},
                    allowed=self.kb.safe,
                    cave_geo=self.kb.cave_geo,
                )
        else:
            target = self._choose_frontier_by_risk(state)
            if target is not None and self.allow_risky_entry:
                self.plan = self._build_frontier_entry_plan(state, target)

        if not self.plan and state.has_arrow and self.kb.possible_wumpus:
            self.plan = plan_shot(
                start_pos=state.pos,
                start_dir=state.dir,
                possible_wumpus=set(self.kb.possible_wumpus.keys()),
                allowed=self.kb.safe,
                cave_geo=self.kb.cave_geo,
            )

        # If every policy branch failed to produce a plan, head home instead of
        # issuing an ineffective CLIMB from an arbitrary cell.
        if not self.plan:
            route_home = plan_route(
                start_pos=state.pos,
                start_dir=state.dir,
                goals={self.start},
                allowed=self.kb.safe,
                cave_geo=self.kb.cave_geo,
            )
            self.plan = route_home + deque([Action.CLIMB])

        return self.plan.popleft() if self.plan else Action.CLIMB

    def _choose_frontier_by_risk(self, state: AgentState) -> CellCoord | None:
        frontier = self.kb.frontier_cells()
        if not frontier:
            self.last_frontier_scores = []
            return None

        # print(f"Frontier cells: {frontier}")
        # print(f"Risk threshold: {self.risk_threshold}")

        scores: list[dict] = []
        for cell in frontier:
            route, staging = plan_route_to_adjacent_staging_cell(
                start_pos=state.pos,
                start_dir=state.dir,
                target_cell=cell,
                allowed=self.kb.safe,
                cave_geo=self.kb.cave_geo,
            )
            if staging is None:
                # print(f"{cell} -> unreachable staging")
                continue

            features = extract_cell_features(
                kb=self.kb,
                candidate=cell,
                agent_pos=state.pos,
                has_arrow=state.has_arrow,
                start=self.start,
            )
            pit_prob = self.pit_model.predict_proba_from_dict(features)
            if pit_prob >= self.risk_threshold:
                continue
            if self.kb.is_known_wumpus(cell):
                continue

            if cell in self.kb.possible_wumpus and not self.kb.wumpus_dead:
                continue

            wumpus_prob = 0.0

            risk = combined_risk(pit_prob, wumpus_prob)
            utility = utility_score(pit_prob, wumpus_prob, len(route))

            # print(
            #     f"{cell} -> pit={pit_prob:.3f}, "
            #     f"wumpus={wumpus_prob:.3f}, "
            #     f"risk={risk:.3f}, "
            #     f"path_cost={len(route)}, "
            #     f"utility={utility:.3f}"
            # )

            scores.append(
                {
                    "cell": cell,
                    "pit_prob": pit_prob,
                    "wumpus_prob": wumpus_prob,
                    "risk": risk,
                    "path_cost": len(route),
                    "utility": utility,
                }
            )

        scores.sort(key=lambda s: (s["risk"], s["path_cost"], -s["utility"]))
        self.last_frontier_scores = scores
        if not scores:
            return None

        best = scores[0]
        # print(f"Best frontier: {best}")

        if best["risk"] <= self.risk_threshold and best["utility"] >= self.utility_floor:
            return best["cell"]
        return None

    def _build_frontier_entry_plan(self, state: AgentState, target: CellCoord) -> deque[Action]:
        route_to_stage, staging = plan_route_to_adjacent_staging_cell(
            start_pos=state.pos,
            start_dir=state.dir,
            target_cell=target,
            allowed=self.kb.safe,
            cave_geo=self.kb.cave_geo,
        )
        if staging is None:
            return deque()

        nav_after = simulate_nav_actions(
            start_pos=state.pos,
            start_dir=state.dir,
            actions=route_to_stage,
            cave_geo=self.kb.cave_geo,
        )
        target_dir = direction_toward(staging, target)
        turn_plan = turns_to_face(nav_after.dir, target_dir)

        full_plan = deque(route_to_stage)
        full_plan.extend(turn_plan)
        full_plan.append(Action.MOVE_FORWARD)
        return full_plan
