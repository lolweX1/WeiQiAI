import numpy as np
import json
import os

class WeiAI:
    def __init__(self, data_file, board):
        self.data_file = data_file
        self.board = board
        self.old_board = [[0 * 19] * 19]
    def update_board(self, board):
        self.old_board = self.board
        self.board = board

# 64 is used because binary, 16 is too small, 128 is too big

def rules(board, old_board):
    """
        return a 19x19 board of 1s and 0s, 1s representing legal moves
    """
    from copy import deepcopy
    from collections import deque
    
    # Constants from main.py
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    BOARD_SIZE = 19
    
    # Determine current player by counting stones
    black_count = np.sum(board == BLACK)
    white_count = np.sum(board == WHITE)
    current_player = BLACK if black_count <= white_count else WHITE
    opponent = WHITE if current_player == BLACK else BLACK
    
    # Helper functions (copied from main.py logic)
    def neighbors(r, c):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                yield nr, nc
    
    def get_group(board_state, r, c):
        color = board_state[r][c]
        visited = set()
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in neighbors(cr, cc):
                if (nr, nc) not in visited and board_state[nr][nc] == color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited
    
    def get_liberties(board_state, group):
        liberties = set()
        for (r, c) in group:
            for nr, nc in neighbors(r, c):
                if board_state[nr][nc] == EMPTY:
                    liberties.add((nr, nc))
        return liberties
    
    def has_liberty(board_state, r, c):
        group = get_group(board_state, r, c)
        return bool(get_liberties(board_state, group))
    
    def capture_dead_groups(board_state, opponent, placed_r, placed_c):
        captured = set()
        for nr, nc in neighbors(placed_r, placed_c):
            if board_state[nr][nc] == opponent and (nr, nc) not in captured:
                group = get_group(board_state, nr, nc)
                if not get_liberties(board_state, group):
                    captured |= group
        for (r, c) in captured:
            board_state[r][c] = EMPTY
        return captured
    
    def board_key(board_state):
        return tuple(board_state[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    
    # Create legal moves board
    legal_moves = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    old_board_key = board_key(old_board) if old_board is not None else None
    
    # Check each position
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            # Position must be empty
            if board[r][c] != EMPTY:
                legal_moves[r][c] = 0
                continue
            
            # Test placement
            test_board = np.array(board, copy=True, dtype=int)
            test_board[r][c] = current_player
            
            # Capture opponent groups
            capture_dead_groups(test_board, opponent, r, c)
            
            # Check suicide (no liberties after capture)
            if not has_liberty(test_board, r, c):
                legal_moves[r][c] = 0
                continue
            
            # Check ko rule
            if old_board_key and board_key(test_board) == old_board_key:
                legal_moves[r][c] = 0
                continue
            
            # Move is legal
            legal_moves[r][c] = 1
    
    return legal_moves

def convert_board_to_token(board):
    """
    board should be 19x19
    this function uses im2col 
    It takes in all 3x3 in the board and coverts it to 1x9 matrices
    It also uses padding sulting in 361 unique 3x3, so this function returns a 361 x 9 matrix
    """
    padded_board = np.pad(board, ((1, 1), (1, 1)), mode='constant')
    rv = np.empty((361, 9), dtype=float)
    cell = 0
    row = 0
    col = 0
    for count in range(361):
        local_index = 0
        # for r in range(row, row + 3):
        #     for c in range(col, col + 3):
        #         rv[count, local_index] = board[r][c]
        #         local_index += 1
        rv[count] = padded_board[row:row+3, col:col+3].flatten()
        col += 1
        if (col > 18):
            col = 0
            row += 1
    return rv

convert_board_to_token("hi")

def store_matrix(json_file, matrix_name, matrix):
    """
    Saves or updates a matrix in a JSON file.
    :param json_file: Path to the .json file
    :param matrix_name: The key (e.g., "layer1_weights")
    :param matrix: The numpy array or list to store
    """
    # 1. Convert NumPy array to a standard Python list so JSON can read it
    if hasattr(matrix, "tolist"):
        matrix_to_save = matrix.tolist()
    else:
        matrix_to_save = matrix

    # 2. Check if the file already exists to avoid overwriting other stored matrices
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # 3. Add or update the specific matrix
    data[matrix_name] = matrix_to_save

    # 4. Write the updated dictionary back to the file
    with open(json_file, 'w') as f:
        json.dump(data, f)
    
    print(f"Successfully stored '{matrix_name}' in {json_file}")

def onion(matrix_1, matrix_2, size):
    new_matrix = matrix_1 @ matrix_2

    reLU = np.maximum(0, new_matrix) # converts all negatives to 0

    tensor_3d = reLU.reshape(size, size, 64)

    padded = np.pad(tensor_3d, ((1, 1), (1, 1), (0, 0)), mode='constant')
    """
    The np.pad width parameter ((1, 1), (1, 1), (0, 0)) applies padding 
    to a 3D array, adding one layer before/after the first dimension (axis 0), 
    one layer before/after the second dimension (axis 1), and zero padding to the 
    third dimension (axis 2)
    """

    matrix_2d = []
    
    # Slide the window
    for r in range(size):
        for c in range(size):
            # 1. Extract the 3x3x64 chunk
            patch = padded[r:r+3, c:c+3, :]
            
            # 2. Flatten that chunk into a single long line of numbers
            flat_patch = patch.flatten()
            
            # 3. Add it as a row to our matrix
            matrix_2d.append(flat_patch)
            
    return np.array(matrix_2d)

    # converting 3d tensor into 3x3x64 chunks and flattening

def rule_reinforcement(board, old_board, move):
    rule_matrix = rules(board, old_board)
    # Flatten move coordinates to 1D index (row * 19 + col)
    if isinstance(move, list) and len(move) == 2:
        move_index = move[0] * 19 + move[1]
    else:
        return None  # Invalid move format
    
    # Return the legality of the move (1 if legal, 0 if illegal)
    return rule_matrix.flatten()[move_index]

def complete_onion(token, matrices = []):
    size = 19
    input_matrix = onion(token, matrices[0], size)
    for ma in matrices[1:]:
        input_matrix = onion(input_matrix, ma, size) # 361 x 576
    return input_matrix

def convert_result_to_probability_board(board, old_board, result):
    # 1. Get the 361 scores
    flat_scores = (result @ policy_weights).flatten()

    # 2. Mask illegal moves (The -1e9 trick)
    legal_mask = rules(board, old_board).flatten()
    flat_scores[legal_mask == 0] = -1e9

    # 3. Add the Pass Score (Index 361)
    # IMPORTANT: Initialize pass_weight ONCE at the top of your script, not here!
    global_average = np.mean(result, axis=0) 
    pass_score = global_average @ pass_weight # Result is a 1-element array
    
    # total_logits now has 362 elements
    total_logits = np.append(flat_scores, pass_score)

    # 4. Softmax on ALL 362 options
    shifted_exp = np.exp(total_logits - np.max(total_logits))
    probabilities = shifted_exp / shifted_exp.sum()

    # 5. Choice out of 362
    move_index = np.random.choice(362, p=probabilities)

    # 6. Return coordinates OR a "Pass" signal
    if move_index == 361:
        return "PASS"
    
    return [move_index // 19, move_index % 19]

rng = np.random.default_rng()
depth_1_matrix = rng.standard_normal((9, 64)) * np.sqrt(2/9) # He initialization
depth_2_matrix = rng.standard_normal((576, 64)) * np.sqrt(2/576) # 576 becase 3 x 3 x 64
depth_3_matrix = rng.standard_normal((576, 64)) * np.sqrt(2/576)
depth_4_matrix = rng.standard_normal((576, 64)) * np.sqrt(2/576)
policy_weights = rng.standard_normal((64, 1)) * np.sqrt(2/64)
pass_weight = pass_weight = rng.standard_normal((64, 1))
