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

pad = Paddle(paddle_x, paddle_y)
ball = Ball(ball_x, ball_y, screen)
bricks = Bricks(screen, brick_width, brick_height)

score = ScoreBoard(10, "white", screen)
score.set_high_score()

running = True
while running:
    screen.fill(BG_COLOR)
    score.show_scores()
    pad.appear(screen)

    if score.is_game_over():
        score.game_over()

    elif bricks.bricks_left() == 0:
        score.success()

    else:
        if bricks.invade_update(pad.rect):
            score.trials = 0

        ball.move()

        if ball.check_for_contact_on_x():
            score.reset_combo()
        if ball.check_for_contact_on_y():
            score.reset_combo()

        # robust paddle hit (ball rect vs paddle rect)
        ball_rect = pg.Rect(int(ball.x - ball.radius), int(ball.y - ball.radius),
                            ball.radius * 2, ball.radius * 2)

        if ball_rect.colliderect(pad.rect) and ball.y_speed > 0:
            ball.bounce_from_paddle(pad.rect)
            ball.y = pad.rect.top - ball.radius - 1
            score.reset_combo()

        if bricks.hit_by_ball(ball.x, ball.y, ball.radius):
            ball.bounce_y()
            score.brick_hit()

        if ball.y + ball.radius >= HEIGHT:
            score.trials -= 1
            score.score = 0
            score.reset_combo()

            bricks.reset_all()

            ball.x = ball_x
            ball.y = pad.rect.top - ball.radius - 1

            pg.time.delay(500)
            ball.y_speed = -abs(ball.y_speed)

    bricks.show_bricks()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        if event.type == pg.KEYDOWN and event.key == pg.K_0:
            score.score = 0
            score.trials = 2
            score.reset_combo()

            bricks.reset_all()

            ball.x = ball_x
            ball.y = ball_y
            ball.y_speed = -abs(ball.y_speed)

    keys = pg.key.get_pressed()
    if keys[pg.K_RIGHT]:
        pad.move_right()
    if keys[pg.K_LEFT]:
        pad.move_left()

    pg.display.flip()
    clock.tick(60)

pg.quit()