import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk

# --- CONSTANTS ---
EMPTY = 0
BLACK = 1
WHITE = -1
BOARD_SIZE = 19
KOMI = 7.5


# --- AI CLASS ---
class WeiAI:
    def __init__(self, data_file, save_every=10):
        self.data_file = data_file
        self.learning_rate = 0.0001
        self.save_every = save_every       # Only save weights every N games
        self.games_since_save = 0
        self.memory = {
            "activations": [],
            "actions": [],
            "probs": [],
            "layer_inputs": [],
            "relu_masks": []
        }

        self.depth_1_matrix = self._init_weight((9, 64))
        self.depth_2_matrix = self._init_weight((576, 64))
        self.depth_3_matrix = self._init_weight((576, 64))
        self.depth_4_matrix = self._init_weight((576, 64))
        self.policy_weights = self._init_weight((576, 1))
        self.pass_weight    = self._init_weight((576, 1))

        self.load_weights()

    def _init_weight(self, shape):
        return np.random.standard_normal(shape) * np.sqrt(2 / shape[0])

    def load_weights(self):
        if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
            with open(self.data_file, 'r') as f:
                try:
                    data = json.load(f)
                    self.depth_1_matrix = np.array(data["d1"])
                    self.depth_2_matrix = np.array(data["d2"])
                    self.depth_3_matrix = np.array(data["d3"])
                    self.depth_4_matrix = np.array(data["d4"])
                    self.policy_weights = np.array(data["policy"])
                    self.pass_weight    = np.array(data["pass"])
                except Exception:
                    pass

    def save_weights(self):
        data = {
            "d1": self.depth_1_matrix.tolist(),
            "d2": self.depth_2_matrix.tolist(),
            "d3": self.depth_3_matrix.tolist(),
            "d4": self.depth_4_matrix.tolist(),
            "policy": self.policy_weights.tolist(),
            "pass":   self.pass_weight.tolist()
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f)

    # ------------------------------------------------------------------
    # OPTIMIZED forward pass
    # Key changes:
    #   • convert_board_to_token uses stride tricks (no Python loop)
    #   • Each conv-like layer uses a single @ then reshape+pad in NumPy
    #   • legal_mask passed in from the game (computed once per move)
    # ------------------------------------------------------------------
    def get_move(self, game, legal_mask_flat):
        token = self._board_to_token_fast(game.board_np)
        matrices = [self.depth_1_matrix, self.depth_2_matrix,
                    self.depth_3_matrix, self.depth_4_matrix]

        current_in = token
        layer_ins, masks = [], []

        for m in matrices:
            layer_ins.append(current_in)
            z    = current_in @ m                          # (361, 64)
            mask = (z > 0)
            masks.append(mask)
            reLU = np.where(mask, z, 0)                   # faster than np.maximum
            # Vectorised 3×3 neighbour gather with stride tricks
            tensor_3d = reLU.reshape(19, 19, 64)
            padded    = np.pad(tensor_3d, ((1,1),(1,1),(0,0)), mode='constant')
            # Use stride tricks to build (361, 9, 64) view then reshape
            current_in = self._sliding_windows(padded)    # (361, 576)

        flat_scores = (current_in @ self.policy_weights).ravel()   # (361,)
        flat_scores[legal_mask_flat == 0] = -1e9

        pass_score = (
            (np.mean(current_in, axis=0) @ self.pass_weight)[0]
            if game.turn_count >= 200 else -1e9
        )

        logits = np.empty(362)
        logits[:361] = flat_scores
        logits[361]  = pass_score
        logits -= logits.max()
        np.exp(logits, out=logits)
        logits /= logits.sum()               # in-place softmax

        move_idx = np.random.choice(362, p=logits)

        self.memory["layer_inputs"].append(layer_ins)
        self.memory["relu_masks"].append(masks)
        self.memory["activations"].append(current_in)
        self.memory["actions"].append(move_idx)
        self.memory["probs"].append(logits.copy())

        if move_idx == 361:
            return "pass"
        return (move_idx % 19, move_idx // 19)   # (col, row)

    @staticmethod
    def _board_to_token_fast(board_np):
        """Convert 19×19 board to (361, 9) token array using stride tricks."""
        padded = np.pad(board_np, 1, mode='constant')          # (21, 21)
        # shape (19,19,3,3) view
        s = padded.strides
        patches = np.lib.stride_tricks.as_strided(
            padded,
            shape=(19, 19, 3, 3),
            strides=(s[0], s[1], s[0], s[1])
        )
        return patches.reshape(361, 9).astype(np.float32)

    @staticmethod
    def _sliding_windows(padded):
        """Build (361, 576) array from a (21,21,64) padded feature map."""
        s = padded.strides
        patches = np.lib.stride_tricks.as_strided(
            padded,
            shape=(19, 19, 3, 3, 64),
            strides=(s[0], s[1], s[0], s[1], s[2])
        )
        return patches.reshape(361, 576).astype(np.float32)

    # ------------------------------------------------------------------
    # OPTIMIZED apply_loss  – vectorised over time steps
    # ------------------------------------------------------------------
    def apply_loss(self, game_result, game_turn, captures):
        actions = self.memory["actions"]
        if not actions:
            return

        move_incentive = 0.5
        capture_bonus  = 2.0 * captures
        pass_reward    = (-50.0 if game_turn < 200
                         else max(-50.0, -50.0 * (1.0 - (game_turn - 200) / 600.0)))

        board_reward = game_result + move_incentive + capture_bonus

        g_policy = np.zeros_like(self.policy_weights)
        g_pass   = np.zeros_like(self.pass_weight)

        activations = self.memory["activations"]
        for t, action_idx in enumerate(actions):
            if action_idx < 361:
                g_policy += activations[t][action_idx, :].reshape(-1, 1) * board_reward
            else:
                g_pass += np.mean(activations[t], axis=0).reshape(-1, 1) * pass_reward

        self.policy_weights += self.learning_rate * g_policy
        self.pass_weight    += self.learning_rate * g_pass

        # Periodic save instead of every game
        self.games_since_save += 1
        if self.games_since_save >= self.save_every:
            self.save_weights()
            self.games_since_save = 0

        self.memory = {k: [] for k in self.memory}


# --- GAME LOGIC ---
class Weiqi:
    # Neighbour offsets as a class constant (avoids re-creation each call)
    _DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self):
        self.board    = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board_np = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player    = BLACK
        self.consecutive_passes = 0
        self.game_over   = False
        self.turn_count  = 0
        self.MAX_TURNS   = 600
        self.black_captures = 0
        self.white_captures = 0
        self.previous_board_hash = None   # Ko rule: track last board state

        # Precompute neighbour lookup: list of lists
        self._nb = [
            [(r+dr, c+dc)
             for dr, dc in self._DELTAS
             if 0 <= r+dr < BOARD_SIZE and 0 <= c+dc < BOARD_SIZE]
            for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        ]

    # ------------------------------------------------------------------
    # OPTIMISED: work with flat 1-D board for BFS (fewer list indexings)
    # ------------------------------------------------------------------
    def _get_group_and_liberties(self, idx):
        """Return (group_set_of_idx, liberty_set_of_idx) for flat index."""
        color   = self.board[idx // 19][idx % 19]
        board   = self.board
        visited = set()
        libs    = set()
        stack   = [idx]
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            r, c = i // 19, i % 19
            for nr, nc in self._nb[i]:
                ni = nr * 19 + nc
                v  = board[nr][nc]
                if v == color:
                    if ni not in visited:
                        stack.append(ni)
                elif v == EMPTY:
                    libs.add(ni)
        return visited, libs

    def count_liberties(self, row, col):
        _, libs = self._get_group_and_liberties(row * 19 + col)
        return len(libs)

    def get_group(self, row, col):
        group, _ = self._get_group_and_liberties(row * 19 + col)
        return {(i // 19, i % 19) for i in group}

    # ------------------------------------------------------------------
    # OPTIMISED: only check neighbours of the last placed stone
    # ------------------------------------------------------------------
    def remove_captured_groups(self, col, row, opponent_color):
        """Only scan neighbours of (row,col) for captures — not whole board."""
        captured = 0
        checked  = set()
        for nr, nc in self._nb[row * 19 + col]:
            if self.board[nr][nc] == opponent_color:
                idx = nr * 19 + nc
                if idx in checked:
                    continue
                group, libs = self._get_group_and_liberties(idx)
                checked.update(group)
                if not libs:
                    captured += len(group)
                    for i in group:
                        gr, gc = i // 19, i % 19
                        self.board[gr][gc] = EMPTY
                        self.board_np[gr, gc] = EMPTY
        if self.current_player == BLACK:
            self.black_captures += captured
        else:
            self.white_captures += captured

    def is_legal_move(self, col, row):
        if self.board[row][col] != EMPTY:
            return False
        self.board[row][col]    = self.current_player
        self.board_np[row, col] = self.current_player
        # Legal if it captures something OR has liberties after placement
        opp = -self.current_player
        legal = False
        for nr, nc in self._nb[row * 19 + col]:
            v = self.board[nr][nc]
            if v == opp and self.count_liberties(nr, nc) == 0:
                legal = True
                break
        if not legal:
            legal = self.count_liberties(row, col) > 0
        # Ko rule: reject if this move would recreate the previous board state
        if legal and self.previous_board_hash is not None:
            test_np = self.board_np.copy()
            for nr, nc in self._nb[row * 19 + col]:
                if self.board[nr][nc] == opp and self.count_liberties(nr, nc) == 0:
                    group, _ = self._get_group_and_liberties(nr * 19 + nc)
                    for i in group:
                        test_np[i // 19, i % 19] = EMPTY
            if hash(test_np.tobytes()) == self.previous_board_hash:
                legal = False
        self.board[row][col]    = EMPTY
        self.board_np[row, col] = EMPTY
        return legal

    # ------------------------------------------------------------------
    # OPTIMISED: compute legal mask once per turn, pass to AI
    # ------------------------------------------------------------------
    def get_legal_mask_flat(self):
        mask = np.zeros(361, dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY and self.is_legal_move(c, r):
                    mask[r * 19 + c] = 1.0
        return mask

    def move(self, col, row=None):
        if self.turn_count >= self.MAX_TURNS:
            self.game_over = True
            return False
        if col == "pass":
            self.consecutive_passes += 1
            if self.consecutive_passes >= 2:
                self.game_over = True
            # Passes don't change board state; keep previous_board_hash as-is
        else:
            if not self.is_legal_move(col, row):
                return False
            # Snapshot current state BEFORE this move (becomes "previous" for next move)
            self.previous_board_hash = hash(self.board_np.tobytes())
            self.board[row][col]    = self.current_player
            self.board_np[row, col] = self.current_player
            self.remove_captured_groups(col, row, -self.current_player)
            self.consecutive_passes = 0
        self.turn_count    += 1
        self.current_player *= -1
        return True

    def get_scores(self):
        b = int(np.sum(self.board_np == BLACK))
        w = int(np.sum(self.board_np == WHITE))
        return b, w + KOMI


# --- GUI ---
class WeiqiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WeiAI Speed Trainer")
        self.game = Weiqi()
        self.bot1 = WeiAI("AI_data.json")
        self.bot2 = WeiAI("AI_2_data.json")
        self.running_ai  = False
        self.stop_requested = False   # stop flag

        self.show_ui    = tk.BooleanVar(value=True)
        self.num_games  = tk.IntVar(value=1)
        self.save_every = tk.IntVar(value=10)

        # ── Layout ────────────────────────────────────────────────────
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Board canvas
        self.canvas = tk.Canvas(main_frame, width=600, height=600, bg="#DBB072")
        self.canvas.pack(side=tk.LEFT)

        # Right-hand controls column
        controls = tk.Frame(main_frame)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        tk.Label(controls, text="Batch Training",
                 font=("Arial", 12, "bold")).pack(pady=(5, 2))

        tk.Label(controls, text="Number of Games:").pack()
        tk.Entry(controls, textvariable=self.num_games, width=10).pack(pady=3)

        tk.Label(controls, text="Save weights every N games:").pack()
        tk.Entry(controls, textvariable=self.save_every, width=10).pack(pady=3)

        tk.Checkbutton(controls, text="Show Board (Slower)",
                       variable=self.show_ui).pack(pady=3)

        # Start / Stop buttons
        btn_row = tk.Frame(controls)
        btn_row.pack(fill=tk.X, pady=6)
        self.btn_start = tk.Button(btn_row, text="▶  Start",
                                   command=self.start_batch,
                                   bg="#2e7d32", fg="white", font=("Arial", 10, "bold"))
        self.btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        self.btn_stop = tk.Button(btn_row, text="■  Stop",
                                  command=self.request_stop,
                                  bg="#c62828", fg="white", font=("Arial", 10, "bold"),
                                  state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # ── Live stats panel ──────────────────────────────────────────
        stats_frame = tk.LabelFrame(controls, text="Session Stats",
                                    font=("Arial", 9, "bold"), padx=6, pady=6)
        stats_frame.pack(fill=tk.X, pady=(10, 4))

        self.var_progress  = tk.StringVar(value="Game  —  /  —")
        self.var_last_res  = tk.StringVar(value="Last result: —")
        self.var_b_wins    = tk.StringVar(value="Black wins: 0")
        self.var_w_wins    = tk.StringVar(value="White wins: 0")
        self.var_last_b    = tk.StringVar(value="Last B score: —")
        self.var_last_w    = tk.StringVar(value="Last W score: —")
        self.var_status    = tk.StringVar(value="Ready")

        tk.Label(stats_frame, textvariable=self.var_progress,
                 font=("Arial", 11, "bold"), fg="#1565c0").pack(anchor="w")
        tk.Label(stats_frame, textvariable=self.var_last_res,
                 font=("Arial", 10)).pack(anchor="w", pady=(4, 0))
        tk.Label(stats_frame, textvariable=self.var_last_b).pack(anchor="w")
        tk.Label(stats_frame, textvariable=self.var_last_w).pack(anchor="w")
        tk.Frame(stats_frame, height=1, bg="#bdbdbd").pack(fill=tk.X, pady=4)
        tk.Label(stats_frame, textvariable=self.var_b_wins,
                 fg="#212121").pack(anchor="w")
        tk.Label(stats_frame, textvariable=self.var_w_wins,
                 fg="#757575").pack(anchor="w")
        tk.Frame(stats_frame, height=1, bg="#bdbdbd").pack(fill=tk.X, pady=4)
        tk.Label(stats_frame, textvariable=self.var_status,
                 fg="#e65100", font=("Arial", 9, "italic")).pack(anchor="w")

        # ── Scrollable game history log ───────────────────────────────
        log_frame = tk.LabelFrame(controls, text="Game History",
                                  font=("Arial", 9, "bold"), padx=4, pady=4)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_box = tk.Listbox(log_frame, width=22, height=14,
                                  font=("Courier", 8),
                                  yscrollcommand=scrollbar.set,
                                  selectbackground="#e3f2fd")
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_box.yview)

        # Session counters
        self._b_wins = 0
        self._w_wins = 0

    # ── Helpers ───────────────────────────────────────────────────────
    def _reset_session_stats(self):
        self._b_wins = 0
        self._w_wins = 0
        self.log_box.delete(0, tk.END)
        for v in (self.var_progress, self.var_last_res,
                  self.var_last_b, self.var_last_w,
                  self.var_b_wins, self.var_w_wins):
            v.set(v.get().split(":")[0] + ": —")
        self.var_b_wins.set("Black wins: 0")
        self.var_w_wins.set("White wins: 0")
        self.var_status.set("Running…")

    def _update_stats(self, game_num, total, b_score, w_score):
        winner  = "Black ♟" if b_score > w_score else "White ○"
        diff    = abs(b_score - w_score)
        if b_score > w_score:
            self._b_wins += 1
        else:
            self._w_wins += 1

        self.var_progress.set(f"Game  {game_num}  /  {total}")
        self.var_last_res.set(f"Last result: {winner} (+{diff:.1f})")
        self.var_last_b.set(f"Last B score: {b_score:.1f}")
        self.var_last_w.set(f"Last W score: {w_score:.1f}")
        self.var_b_wins.set(f"Black wins: {self._b_wins}")
        self.var_w_wins.set(f"White wins: {self._w_wins}")

        tag = "B" if b_score > w_score else "W"
        entry = f"G{game_num:>4}: {tag}  B{b_score:>6.1f} W{w_score:>6.1f}"
        self.log_box.insert(tk.END, entry)
        self.log_box.see(tk.END)           # auto-scroll to newest
        self.root.update_idletasks()

    # ── Board drawing ─────────────────────────────────────────────────
    def draw_board(self):
        self.canvas.delete("all")
        m, c = 30, 30
        for i in range(19):
            self.canvas.create_line(m + i*c, m, m + i*c, m + 18*c)
            self.canvas.create_line(m, m + i*c, m + 18*c, m + i*c)
        board = self.game.board
        for r in range(19):
            for col in range(19):
                v = board[r][col]
                if v != EMPTY:
                    color = "black" if v == BLACK else "white"
                    x, y = m + col*c, m + r*c
                    self.canvas.create_oval(x-12, y-12, x+12, y+12, fill=color)
        self.root.update_idletasks()

    # ── Button handlers ───────────────────────────────────────────────
    def start_batch(self):
        self.stop_requested = False
        n = self.save_every.get()
        self.bot1.save_every = n
        self.bot2.save_every = n
        self.running_ai = True
        self._reset_session_stats()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.root.after(10, self.run_batch)

    def request_stop(self):
        """Signal the batch loop to stop after the current game finishes."""
        self.stop_requested = True
        self.var_status.set("Stopping after this game…")
        self.btn_stop.config(state=tk.DISABLED)

    # ── Main training loop ────────────────────────────────────────────
    def run_batch(self):
        total      = self.num_games.get()
        show_ui    = self.show_ui.get()
        draw_every = 10

        for i in range(total):
            # Honour stop request between games
            if self.stop_requested:
                self.var_status.set(f"Stopped at game {i} / {total}")
                break

            self.game = Weiqi()
            self.var_progress.set(f"Game  {i+1}  /  {total}")
            if show_ui:
                self.root.update_idletasks()

            move_count = 0
            while not self.game.game_over:
                legal_mask  = self.game.get_legal_mask_flat()
                current_bot = self.bot1 if self.game.current_player == BLACK else self.bot2
                move        = current_bot.get_move(self.game, legal_mask)

                if move == "pass":
                    self.game.move("pass")
                else:
                    self.game.move(move[0], move[1])

                move_count += 1
                if show_ui and move_count % draw_every == 0:
                    self.draw_board()

            b, w = self.game.get_scores()
            diff = b - w
            self.bot1.apply_loss( diff, self.game.turn_count, self.game.black_captures)
            self.bot2.apply_loss(-diff, self.game.turn_count, self.game.white_captures)
            self._update_stats(i + 1, total, b, w)

        else:
            # Loop completed without a break → full batch done
            self.var_status.set("Batch complete ✓")

        # Always save on exit
        self.bot1.save_weights()
        self.bot2.save_weights()

        self.running_ai = False
        self.stop_requested = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if show_ui:
            self.draw_board()


if __name__ == "__main__":
    root = tk.Tk()
    gui = WeiqiGUI(root)
    root.mainloop()
