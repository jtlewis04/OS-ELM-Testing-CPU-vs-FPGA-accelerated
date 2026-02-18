# main.py
from ball import Ball
import pygame as pg
from bricks import Bricks
from paddle import Paddle
from settings import *
from scoreboard import ScoreBoard

pg.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Breakout Game")
clock = pg.time.Clock()

# OBJECTS
pad = Paddle(paddle_x, paddle_y)
ball = Ball(ball_x, ball_y, screen)
bricks = Bricks(screen, brick_width, brick_height)

score = ScoreBoard(10, "white", screen)
score.set_high_score()   # <-- IMPORTANT (parentheses)

running = True
while running:
    screen.fill(BG_COLOR)
    score.show_scores()
    pad.appear(screen)
    score.set_high_score() 

    # ---------------------------
    # GAME OVER / WIN CHECKS
    # ---------------------------
    if score.is_game_over():
        score.game_over()

    elif len(bricks.bricks) == 0:
        score.success()

    else:
        
        if bricks.invade_update(pad.rect):
            score.trials = 0  # triggers game over state

        
        ball.move()

        # bounce off walls
        ball.check_for_contact_on_x()
        ball.check_for_contact_on_y()

        # ball hits paddle
        if (pad.rect.y < ball.y + ball.radius < pad.rect.y + pad.height
                and pad.rect.x < ball.x + ball.radius < pad.rect.x + pad.width):
            ball.bounce_y()
            ball.y = pad.y - ball.radius

        # ball hits brick
        for brick in bricks.bricks[:]:
            if (brick.collidepoint(ball.x, ball.y - ball.radius)
                    or brick.collidepoint(ball.x, ball.y + ball.radius)):
                bricks.bricks.remove(brick)
                ball.bounce_y()
                score.score += 1

        # ball hits bottom
        if ball.y + ball.radius >= HEIGHT:
            ball.y = pad.y - ball.radius
            pg.time.delay(500)
            score.trials -= 1
            ball.bounce_y()

    bricks.show_bricks()

    # quit
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    # input
    keys = pg.key.get_pressed()
    if keys[pg.K_RIGHT]:
        pad.move_right()
    if keys[pg.K_LEFT]:
        pad.move_left()

    if keys[pg.K_0]:
        score.score = 0
        score.trials = 2

        bricks.bricks.clear()
        bricks.brick_colors.clear()
        bricks.set_values()

        bricks.reset_invade()

        ball.x = ball_x
        ball.y = ball_y



    pg.display.flip()
    clock.tick(60)

pg.quit()
