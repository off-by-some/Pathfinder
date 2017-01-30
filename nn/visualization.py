import pygame
import random
import math

pygame.init()

BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
YELLOW = (255, 255, 0)
RED = ( 255, 0, 0)
BLUE = 	(0,0,255)
GAINSBORO = (220, 220, 220)
ORANGE = (255, 50, 0)

WIDTH = 800
HEIGHT = 650

# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()


class Button:
   def __init__(self, text):
      self.text = text
      self.is_hover = False
      self.default_color = (100,100,100)
      self.hover_color = (255,255,255)
      self.font_color = (0,0,0)
      self.obj = None

   def label(self):
      '''button label font'''
      font = pygame.font.Font(None, 20)
      return font.render(self.text, 1, self.font_color)

   def color(self):
      '''change color when hovering'''
      if self.is_hover:
         return self.hover_color
      else:
         return self.default_color

   def draw(self, screen, mouse, rectcoord, labelcoord):
      '''create rect obj, draw, and change color based on input'''
      self.obj  = pygame.draw.rect(screen, self.color(), rectcoord)
      screen.blit(self.label(), labelcoord)

      #change color if mouse over button
      self.check_hover(mouse)

   def check_hover(self, mouse):
      '''adjust is_hover value based on mouse over button - to change hover color'''
      if self.obj.collidepoint(mouse):
         self.is_hover = True
      else:
         self.is_hover = False


class Display:
    def __init__(self, level):
        self.resize(level)
        pygame.display.set_caption("pathfinder")
        self.font = pygame.font.SysFont("monospace", 18)
        self.episode = 0
        self.current_reward = 0
        self.mean_reward = 0
        self.show_fog = False
        self.btn = Button("Enable Fog")

        # The action we pressed
        self.action = None

    def resize(self, level):
        level_height = len(level)
        level_width = len(level[0])
        self.level_width = level_width
        self.level_height = level_height
        self.scale = min(WIDTH / level_width, HEIGHT / level_height)
        s_lw = level_width * self.scale
        s_lh = level_height * self.scale
        width = s_lw if s_lw <= WIDTH else WIDTH
        height = s_lh if s_lh < HEIGHT else HEIGHT
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size)

    def draw(self, mouse, level):
        scale_x = WIDTH / len(level[0])
        scale_y =  HEIGHT / len(level)
        block = (scale_x, scale_y)

        # how many blocks we can fit per x
        per_x = WIDTH / scale_x

        # How many blocks per y
        per_y = HEIGHT / scale_y

        # Clear the screen
        self.screen.fill(WHITE)

        # Our offsets on where we are
        x_pos = 0
        y_pos = 0
        for y in range(len(level)):
            line = level[y]
            for x in range(len(level[0])):
                ch = line[x]

                if (ch == "x"):
                    pygame.draw.rect(self.screen, BLACK, [self.scale * x, self.scale * y, self.scale, self.scale], 0)
                if (ch == "G"):
                    pygame.draw.rect(self.screen, random.choice([GREEN, RED, BLUE]), [self.scale * x, self.scale * y, self.scale, self.scale], 0)
                if (ch == "!"):
                    pygame.draw.rect(self.screen, RED, [self.scale * x, self.scale * y, self.scale, self.scale], 0)
                if (ch == "o"):
                    pygame.draw.circle(self.screen, BLACK, (self.scale * x + (self.scale / 2), self.scale * y + (self.scale / 2)), int(self.scale / 1.6), 0)
                    pygame.draw.circle(self.screen, YELLOW, (self.scale * x + (self.scale / 2), self.scale * y + (self.scale / 2)), self.scale / 2, 0)
                if (ch == "@"):
                    pygame.draw.circle(self.screen, BLUE, (self.scale * x + (self.scale / 2), self.scale * y + (self.scale / 2)), self.scale / 2, 0)
                if (ch == "."):
                    pygame.draw.circle(self.screen, GREEN, (self.scale * x + (self.scale / 2), self.scale * y + (self.scale / 2)), self.scale / 6, 0)
                if (ch == ":"):
                    pygame.draw.circle(self.screen, GAINSBORO, (self.scale * x + (self.scale / 2), self.scale * y + (self.scale / 2)), self.scale / 6, 0)


                x_pos += 1
            x_pos = 0
            y_pos += 1

        x = self.player.pos[0]
        y = self.player.pos[1]
        start_pos = (self.scale * x + (self.scale / 2), self.scale * y + (self.scale / 2))
        end_pos = (start_pos[0] + (math.cos(self.player.heading) * 80), start_pos[1] + (math.sin(self.player.heading) * 80))
        pygame.draw.line(self.screen, ORANGE, start_pos, end_pos, 1)
        start_pos = (self.scale * x + (self.scale / 2), self.scale * y + (self.scale / 2))
        end_pos = (start_pos[0] + (math.cos(self.player.heading + self.player.viewing_angle) * 80), start_pos[1] + (math.sin(self.player.heading + self.player.viewing_angle) * 80))
        pygame.draw.line(self.screen, ORANGE, start_pos, end_pos, 1)


        label = self.font.render("Current Episode: %s" % self.episode, 1, RED)
        label2 = self.font.render("Current Reward: %s" % self.current_reward, 1, RED)
        label3 = self.font.render("Mean Reward: %s" % self.mean_reward, 1, RED)
        label4 = self.font.render("Action pressed: %s" % self.action, 1, RED)
        # label5 = self.font.render("Heading: %s" % self.player.heading, 1, RED)

        self.screen.blit(label, (40, 20))
        self.screen.blit(label2, (40, 40))
        self.screen.blit(label3, (40, 60))
        self.screen.blit(label4, (40, 80))
        # self.screen.blit(label5, (40, 100))

        btn_text = "Disable" if self.show_fog else "Enable"

        # Fog of war button
        self.btn = Button("%s Fog" % btn_text)

        self.btn.draw(self.screen, mouse, (40,120,120,20), (55,123))
        self.btn.check_hover(mouse)


    def update(self, new_level):
        self.level = new_level
        self.tick()

    def tick(self):
        mouse = pygame.mouse.get_pos()

        # --- Main event loop
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                  pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.btn.obj.collidepoint(mouse):
                   self.show_fog = not self.show_fog

        # --- Game logic should go here
        self.draw(mouse, self.level)



        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # --- Limit to 60 frames per second
        clock.tick(60)
