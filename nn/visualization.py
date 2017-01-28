import pygame
import random

pygame.init()

BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
YELLOW = (255, 255, 0)
RED = ( 255, 0, 0)
BLUE = 	(0,0,255)
GAINSBORO = (220, 220, 220)

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
        self.size = (WIDTH, HEIGHT)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("pathfinder")
        self.font = pygame.font.SysFont("monospace", 18)
        self.episode = 0
        self.current_reward = 0
        self.mean_reward = 0
        self.show_fog = False
        self.btn = Button("Enable Fog")

        # The action we pressed
        self.action = None

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
        for y in range(per_y):
            line = level[y]
            for x in range(per_x):
                ch = line[x]

                if (ch == "x"):
                    pygame.draw.rect(self.screen, BLACK, [scale_x * x, scale_y * y, block[0], block[1]], 0)
                if (ch == "G"):
                    pygame.draw.rect(self.screen, random.choice([GREEN, RED, BLUE]), [scale_x * x, scale_y * y, block[0], block[1]], 0)
                if (ch == "!"):
                    pygame.draw.rect(self.screen, RED, [scale_x * x, scale_y * y, block[0], block[1]], 0)
                if (ch == "o"):
                    pygame.draw.circle(self.screen, BLACK, (scale_x * x + (scale_x / 2), scale_y * y + (scale_x / 2)), int(scale_x / 1.6), 0)
                    pygame.draw.circle(self.screen, YELLOW, (scale_x * x + (scale_x / 2), scale_y * y + (scale_x / 2)), scale_x / 2, 0)
                if (ch == "@"):
                    pygame.draw.circle(self.screen, BLUE, (scale_x * x + (scale_x / 2), scale_y * y + (scale_x / 2)), scale_x / 2, 0)
                if (ch == "."):
                    pygame.draw.circle(self.screen, GREEN, (scale_x * x + (scale_x / 2), scale_y * y + (scale_x / 2)), scale_x / 6, 0)
                if (ch == ":"):
                    pygame.draw.circle(self.screen, GAINSBORO, (scale_x * x + (scale_x / 2), scale_y * y + (scale_x / 2)), scale_x / 6, 0)


                x_pos += 1
            x_pos = 0
            y_pos += 1


        label = self.font.render("Current Episode: %s" % self.episode, 1, RED)
        label2 = self.font.render("Current Reward: %s" % self.current_reward, 1, RED)
        label3 = self.font.render("Mean Reward: %s" % self.mean_reward, 1, RED)
        label4 = self.font.render("Action pressed: %s" % self.action, 1, RED)

        self.screen.blit(label, (40, 20))
        self.screen.blit(label2, (40, 40))
        self.screen.blit(label3, (40, 60))
        self.screen.blit(label4, (40, 80))

        btn_text = "Disable" if self.show_fog else "Enable"

        # Fog of war button
        self.btn = Button("%s Fog" % btn_text)

        self.btn.draw(self.screen, mouse, (40,100,100,20), (55,103))
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
