from collections import deque

from app.agent.KB import KnowledgeBase
from app.agent.agent_state import AgentState
from app.agent.planner import plan_route, plan_shot
from app.domain.action import Action
from app.domain.cell import CellCoord
from app.domain.percept import Percept


class WumpusAgent:
    def __init__(self, kb: KnowledgeBase, start: CellCoord = CellCoord(0, 0)) -> None:
        self.kb = kb
        self.plan = deque()
        self.start = start

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
            self.plan = plan_route(
                start_pos=state.pos,
                start_dir=state.dir,
                goals=safe_unvisited,
                allowed=self.kb.safe,
                cave_geo=self.kb.cave_geo,
            )

        elif state.has_arrow and self.kb.possible_wumpus:
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
