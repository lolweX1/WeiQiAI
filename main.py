"""
Chinese Weiqi (Go) - 19x19
Rules: Chinese scoring with komi, territory, captures, dead stone removal,
       seki detection, ko rule, suicide rule.

Usage:
    game = Weiqi()
    game.move(col, row)   # col and row are 0-indexed (0..18)
    game.move("pass")     # or game.move(None)
    game.score()          # returns final score once game is over
"""

from copy import deepcopy
from collections import deque


EMPTY = 0
BLACK = 1
WHITE = -1
BOARD_SIZE = 19
KOMI = 7.5          # Standard Chinese komi
MIN_STONES_TO_END = 200  # Both-pass only ends game above this stone count


class Weiqi:
    def __init__(self):
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = BLACK          # Black goes first
        self.captures = {BLACK: 0, WHITE: 0}
        self.history = []                    # board snapshots (for ko)
        self.consecutive_passes = 0
        self.game_over = False
        self.move_log = []
        self.dead_stones: set = set()        # (r, c) pairs marked dead

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def move(self, col, row=None):
        """
        Place a stone or pass.

            game.move(col, row)   -> place stone, 0-indexed
            game.move("pass")     -> pass
            game.move(None)       -> pass
        """
        if self.game_over:
            print("Game is already over. Call game.score() to see results.")
            return False

        # --- Detect pass ---
        if col == "pass" or col is None or (isinstance(col, str) and col.lower() == "pass"):
            return self._do_pass()

        # --- Validate coordinates ---
        if not (0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE):
            print(f"Invalid coordinates ({col}, {row}). Must be 0-{BOARD_SIZE - 1}.")
            return False

        if self.board[row][col] != EMPTY:
            print(f"({col}, {row}) is already occupied.")
            return False

        # --- Attempt placement ---
        new_board = deepcopy(self.board)
        new_board[row][col] = self.current_player

        opponent = WHITE if self.current_player == BLACK else BLACK
        captured = self._capture_dead_groups(new_board, opponent, row, col)

        # Suicide check
        if not self._has_liberty(new_board, row, col):
            print(f"Illegal move at ({col}, {row}): suicide.")
            return False

        # Ko check
        board_key = self._board_key(new_board)
        if self.history and board_key == self.history[-1]:
            print(f"Illegal move at ({col}, {row}): Ko rule violation.")
            return False

        # --- Commit ---
        self.history.append(self._board_key(self.board))
        self.board = new_board
        self.captures[self.current_player] += len(captured)
        self.consecutive_passes = 0

        player_name = "Black" if self.current_player == BLACK else "White"
        self.move_log.append(f"{player_name} plays ({col}, {row}), captures {len(captured)}")
        print(self.move_log[-1])

        self._switch_player()

        # End game if no legal moves remain for either player
        if not self._has_legal_moves(self.current_player) and \
           not self._has_legal_moves(WHITE if self.current_player == BLACK else BLACK):
            stone_count = self._stone_count()
            self.game_over = True
            print(f"No legal moves remain for either player ({stone_count} stones). Game over. Call game.score().")

        return True

    def score(self, dead_stones=None):
        """
        Compute Chinese-style score. No dead stone input required.

        Parameters
        ----------
        dead_stones : list of (col, row) tuples, optional
            Override dead stones manually if needed.

        Returns
        -------
        dict with full scoring breakdown.
        """
        if dead_stones is not None:
            self.dead_stones = {(row, col) for (col, row) in dead_stones}

        scoring_board = deepcopy(self.board)
        dead_by_color = {BLACK: 0, WHITE: 0}
        for (r, c) in self.dead_stones:
            color = scoring_board[r][c]
            if color != EMPTY:
                dead_by_color[color] += 1
                scoring_board[r][c] = EMPTY

        stones = {BLACK: 0, WHITE: 0}
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if scoring_board[r][c] == BLACK:
                    stones[BLACK] += 1
                elif scoring_board[r][c] == WHITE:
                    stones[WHITE] += 1

        territory, seki_points = self._compute_territory(scoring_board)

        black_score = stones[BLACK] + territory[BLACK]
        white_score = stones[WHITE] + territory[WHITE] + KOMI

        if black_score > white_score:
            winner = f"Black wins by {black_score - white_score:.1f}"
        elif white_score > black_score:
            winner = f"White wins by {white_score - black_score:.1f}"
        else:
            winner = "Draw (jigo)"

        detail = {
            "black_stones_alive": stones[BLACK],
            "white_stones_alive": stones[WHITE],
            "black_territory": territory[BLACK],
            "white_territory": territory[WHITE],
            "seki_points": seki_points,
            "komi": KOMI,
            "black_score": black_score,
            "white_score": white_score,
            "winner": winner,
            "dead_stones_removed": dead_by_color,
        }

        print("\n=== FINAL SCORE ===")
        print(f"  Black: {stones[BLACK]} stones + {territory[BLACK]} territory = {black_score}")
        print(f"  White: {stones[WHITE]} stones + {territory[WHITE]} territory + {KOMI} komi = {white_score}")
        print(f"  Seki points (neutral, not scored): {seki_points}")
        print(f"  Dead stones removed: Black={dead_by_color[BLACK]}, White={dead_by_color[WHITE]}")
        print(f"  {winner}")
        return detail

    def mark_dead(self, col, row):
        """Manually mark a stone group at (col, row) as dead."""
        r, c = row, col
        if self.board[r][c] == EMPTY:
            print("No stone there.")
            return
        group = self._get_group(self.board, r, c)
        for (gr, gc) in group:
            self.dead_stones.add((gr, gc))
        print(f"Marked {len(group)} stones as dead.")

    def print_board(self):
        """Print current board state."""
        symbols = {EMPTY: ".", BLACK: "●", WHITE: "○"}
        col_labels = "ABCDEFGHJKLMNOPQRST"
        print("   " + " ".join(col_labels[:BOARD_SIZE]))
        for r in range(BOARD_SIZE):
            row_label = str(BOARD_SIZE - r).rjust(2)
            print(row_label + " " + " ".join(symbols[self.board[r][c]] for c in range(BOARD_SIZE)))
        print()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _do_pass(self):
        player_name = "Black" if self.current_player == BLACK else "White"
        self.consecutive_passes += 1
        self.move_log.append(f"{player_name} passes")
        print(self.move_log[-1])
        self.history.append(self._board_key(self.board))
        if self.consecutive_passes >= 2:
            self._check_end_on_double_pass()
        self._switch_player()
        return True

    def _check_end_on_double_pass(self):
        """End the game on double-pass only if enough stones are on the board,
        OR if there are genuinely no legal moves left for either player."""
        stone_count = self._stone_count()
        no_moves = (
            not self._has_legal_moves(BLACK) and
            not self._has_legal_moves(WHITE)
        )

        if no_moves:
            self.game_over = True
            print(f"No legal moves remain for either player ({stone_count} stones). Game over. Call game.score().")
        elif stone_count >= MIN_STONES_TO_END:
            self.game_over = True
            print(f"Both players passed ({stone_count} stones on board). Game over. Call game.score().")
        else:
            print(
                f"Both players passed, but only {stone_count} stones on board "
                f"(minimum {MIN_STONES_TO_END} required to end by passing) — game continues."
            )
            self.consecutive_passes = 0

    def _stone_count(self):
        return sum(
            1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            if self.board[r][c] != EMPTY
        )

    def _has_legal_moves(self, player):
        """Return True if `player` has at least one legal move on the current board."""
        opponent = WHITE if player == BLACK else BLACK
        prev_key = self.history[-1] if self.history else None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != EMPTY:
                    continue
                test = deepcopy(self.board)
                test[r][c] = player
                self._capture_dead_groups(test, opponent, r, c)
                if not self._has_liberty(test, r, c):
                    continue  # suicide
                if prev_key and self._board_key(test) == prev_key:
                    continue  # ko
                return True
        return False

    def _switch_player(self):
        self.current_player = WHITE if self.current_player == BLACK else BLACK

    def _board_key(self, board):
        return tuple(board[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))

    def _neighbors(self, r, c):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                yield nr, nc

    def _get_group(self, board, r, c):
        color = board[r][c]
        visited = set()
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in self._neighbors(cr, cc):
                if (nr, nc) not in visited and board[nr][nc] == color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    def _get_liberties(self, board, group):
        liberties = set()
        for (r, c) in group:
            for nr, nc in self._neighbors(r, c):
                if board[nr][nc] == EMPTY:
                    liberties.add((nr, nc))
        return liberties

    def _has_liberty(self, board, r, c):
        group = self._get_group(board, r, c)
        return bool(self._get_liberties(board, group))

    def _capture_dead_groups(self, board, opponent, placed_r, placed_c):
        captured = set()
        for nr, nc in self._neighbors(placed_r, placed_c):
            if board[nr][nc] == opponent and (nr, nc) not in captured:
                group = self._get_group(board, nr, nc)
                if not self._get_liberties(board, group):
                    captured |= group
        for (r, c) in captured:
            board[r][c] = EMPTY
        return captured

    def _compute_territory(self, board):
        visited = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        territory = {BLACK: 0, WHITE: 0}
        seki_points = 0

        seki_empty = self._find_seki_points(board)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == EMPTY and not visited[r][c] and (r, c) not in seki_empty:
                    region = set()
                    borders = set()
                    queue = deque([(r, c)])
                    visited[r][c] = True
                    region.add((r, c))
                    while queue:
                        cr, cc = queue.popleft()
                        for nr, nc in self._neighbors(cr, cc):
                            if board[nr][nc] == EMPTY and not visited[nr][nc] and (nr, nc) not in seki_empty:
                                visited[nr][nc] = True
                                region.add((nr, nc))
                                queue.append((nr, nc))
                            elif board[nr][nc] != EMPTY:
                                borders.add(board[nr][nc])
                    if len(borders) == 1:
                        owner = borders.pop()
                        territory[owner] += len(region)
                elif (r, c) in seki_empty:
                    visited[r][c] = True
                    seki_points += 1

        return territory, seki_points

    def _find_seki_points(self, board):
        seki_empty = set()
        checked_groups = set()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = board[r][c]
                if color == EMPTY:
                    continue
                group = frozenset(self._get_group(board, r, c))
                if group in checked_groups:
                    continue
                checked_groups.add(group)

                liberties = self._get_liberties(board, group)
                opponent = WHITE if color == BLACK else BLACK

                adj_opp_groups = set()
                for (lr, lc) in liberties:
                    for nr, nc in self._neighbors(lr, lc):
                        if board[nr][nc] == opponent:
                            adj_opp_groups.add(frozenset(self._get_group(board, nr, nc)))

                for opp_group in adj_opp_groups:
                    opp_liberties = self._get_liberties(board, opp_group)
                    shared = liberties & opp_liberties
                    if shared == liberties and shared == opp_liberties:
                        seki_empty |= shared

        return seki_empty


# ------------------------------------------------------------------ #
#  Interactive CLI                                                    #
# ------------------------------------------------------------------ #

def play():
    game = Weiqi()
    game.print_board()
    while not game.game_over:
        player = "Black" if game.current_player == BLACK else "White"
        raw = input(f"{player} to move (col row) or 'pass': ").strip()
        if raw.lower() == "pass":
            game.move("pass")
        else:
            try:
                parts = raw.split()
                col, row = int(parts[0]), int(parts[1])
                game.move(col, row)
            except (ValueError, IndexError):
                print("Enter two numbers for col and row, e.g. '3 4', or 'pass'.")
                continue
        game.print_board()

    game.score()


if __name__ == "__main__":
    play()