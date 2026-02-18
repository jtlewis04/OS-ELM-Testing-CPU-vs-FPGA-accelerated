import pygame as pg
from settings import *

class Ball:
    def __init__(self, x, y, screen):
        self.screen = screen
        self.x = x
        self.y = y
        self.radius = ball_radius

        self.x_speed = ball_x_speed
        self.y_speed = ball_y_speed

    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed
        pg.draw.circle(self.screen, ball_color, (int(self.x), int(self.y)), self.radius)

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def check_for_contact_on_x(self):
        if self.x - self.radius <= 0 or self.x + self.radius >= WIDTH:
            self.bounce_x()

    def check_for_contact_on_y(self):
        if self.y - self.radius <= 0:
            self.bounce_y()
