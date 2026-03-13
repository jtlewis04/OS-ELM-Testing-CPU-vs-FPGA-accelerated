import numpy as np
from settings import bricks_per_row, bricks_per_col, WIDTH, HEIGHT, paddle_width

#Encodes game state into a vector to be fed to AI

# ---------------------------------------------------------------------------
# State encoding constants
# ---------------------------------------------------------------------------
# The brick grid starts at start_rows = bricks_per_col // 3 = 5 rows of 10 cols
# but grows over time via invade_update.  We observe a fixed window of the
# bottom K rows closest to the paddle sincee the ball is unlikely to interact with higher rows and
# the AI needs to prioritize closer bricks or risk losing.
BRICK_WINDOW_ROWS = 8
BRICK_COLS = bricks_per_row          # 10

# 4 ball + 1 paddle + 5 relational + BRICK_WINDOW_ROWS*BRICK_COLS brick grid
STATE_DIM = 4 + 1 + 5 + BRICK_WINDOW_ROWS * BRICK_COLS  # = 90


def encode_game(ball, paddle, bricks, score):
    """
    Build a fixed-length float32 feature vector from live game objects.

    Layout:
      [0]    ball_x / WIDTH
      [1]    ball_y / HEIGHT
      [2]    ball x-direction sign  (+1 / -1)
      [3]    ball y-direction sign  (+1 / -1)
      [4]    paddle_x normalized    (0..1)
      [5]    ball-to-paddle horizontal offset / WIDTH
      [6]    ball vertical proximity (ball_y / paddle_y)
      [7]    fraction of bricks remaining
      [8]    combo / combo_max
      [9]    ball base_speed normalized
      [10..] bottom BRICK_WINDOW_ROWS rows of grid, flattened (binary 0/1)
    """
    features = []

    # ball (4)
    features.append(ball.x / WIDTH)
    features.append(ball.y / HEIGHT)
    features.append(1.0 if ball.x_speed >= 0 else -1.0)
    features.append(1.0 if ball.y_speed >= 0 else -1.0)

    # paddle (1)
    max_paddle_x = WIDTH - paddle.width
    features.append(paddle.rect.x / max_paddle_x if max_paddle_x > 0 else 0.5)

    #relational (5)
    pad_cx = paddle.rect.x + paddle.width / 2
    features.append((ball.x - pad_cx) / WIDTH)

    paddle_top_y = paddle.rect.y
    features.append(ball.y / paddle_top_y if paddle_top_y > 0 else 1.0)

    total_possible = bricks.cols * len(bricks.grid) if len(bricks.grid) > 0 else 1
    features.append(bricks.bricks_left() / max(total_possible, 1))

    features.append(score.combo / max(score.combo_max, 1))

    #ball speed
    features.append(ball.base_speed / (ball._min_speed * 1.6) if ball._min_speed > 0 else 1.0)

    #brick grid window (BRICK_WINDOW_ROWS × BRICK_COLS)
    grid = bricks.grid
    num_rows = len(grid)
    for r_offset in range(BRICK_WINDOW_ROWS):
        src = num_rows - BRICK_WINDOW_ROWS + r_offset   # bottom-aligned
        for c in range(BRICK_COLS):
            if 0 <= src < num_rows:
                features.append(float(grid[src][c]))
            else:
                features.append(0.0)

    return np.array(features, dtype=np.float32)