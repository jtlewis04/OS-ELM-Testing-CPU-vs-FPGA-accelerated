# EECE4632-Final-Project
James Lewis and Akshaj Sirineni

## Game Intro
- Paddle moves left/right to bounce a ball into a grid of colored bricks
- Bricks slowly invade downward over time — if they reach the paddle, it's game over
- Combo system rewards consecutive brick hits without touching the paddle
- Ball speed increases with combo via `set_combo_speed(combo)`
- Single-life design: losing the ball ends the game immediately
- Press `0` to restart after a game over or win

## How to Run

### Install Dependencies
- Run `pip install -r requirements.txt`

### Play Manually
- Run `python main.py`
- Use left/right arrow keys to move the paddle

### Play on Xilinx Board (TODO)
- Run game in Jupyter notebook
- Button0 to move left, Button 1 to move right

## State Encoding

### Overview
- The AI algorithm observes the game through a feature vector of 90 floats
- Extracted each frame by `encode_game()` in `encoder.py`

### Feature Breakdown
- **Ball (4 features):** normalized x/y position, x-direction sign, y-direction sign
- **Paddle (1 feature):** normalized x position across the playable width
- **Relational (5 features):**
  - Horizontal offset from ball to paddle center
  - Vertical proximity of ball to paddle
  - Fraction of bricks remaining
  - Current combo normalized against combo max
  - Ball speed normalized against min speed
- **Brick grid (depends on settings):** grid flattened as binary 0/1 values

### Design Decisions
- Ball speed is encoded explicitly because combo-driven acceleration changes the agent's effective reaction time
- Rows beyond the 8-row window are ignored — they are too far from the paddle to influence short-term decisions