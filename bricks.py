import random
import pygame as pg

class Bricks:
    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height
        self.random_colors = ['blue', 'yellow', 'red', 'green', 'orange']
        self.bricks = []
        self.brick_colors = []
        self.set_values()

        self.start_delay_ms = 2000         
        self.spawn_time = pg.time.get_ticks()

        self.invade_speed = 3.0            
        self.invade_max_speed = 25.0       # cap speed
        self.invade_accel = 1.5            # pixels per second^2 

        self.last_update_time = pg.time.get_ticks()

    def set_values(self):
        y_values = [int(y) for y in range(100, 200, 25)]
        x_values = [int(x) for x in range(10, 550, 42)]
        y_index = 0
        self.loop(x_values, y_values, y_index)

    def loop(self, x_values, y_values, y_index):
        for n in x_values:
            if n == x_values[-1]:
                if y_index < len(y_values) - 1:
                    y_index += 1
                    self.loop(x_values, y_values, y_index)
            else:
                x = n
                y = y_values[y_index]
                brick = pg.Rect(x, y, self.width, self.height)
                self.bricks.append(brick)
                self.brick_colors.append(random.choice(self.random_colors))

    def invade_update(self, paddle_rect) -> bool:
        now = pg.time.get_ticks()

        if now - self.spawn_time < self.start_delay_ms:
            self.last_update_time = now
            return False

        dt = (now - self.last_update_time) / 1000.0
        self.last_update_time = now

        if dt > 0.05:
            dt = 0.05

        # Accelerate smoothly
        self.invade_speed = min(self.invade_speed + self.invade_accel * dt, self.invade_max_speed)

        # Move bricks down by speed * time
        dy = self.invade_speed * dt

        for brick in self.bricks:
            brick.y += dy

            # Game over if brick reaches paddle line
            if brick.bottom >= paddle_rect.top:
                return True

        return False

    def reset_invade(self):
        self.spawn_time = pg.time.get_ticks()
        self.last_update_time = pg.time.get_ticks()
        self.invade_speed = 3.0

    def show_bricks(self):
        for i in range(len(self.bricks)):
            brick = self.bricks[i]
            color = self.brick_colors[i]
            pg.draw.rect(self.screen, color, brick)
