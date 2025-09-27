import numpy as np
from random import randint

# Kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.graphics import Color, Line

# Our DQN
from model import DQN

# ========= Window / Config =========
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Load KV so custom classes (Car, Ball1/2/3, root layout) exist
Builder.load_file('car.kv')

# ========= Globals =========
longg, larg = 1200, 800  # width, height of the world

# IMPORTANT: use (height, width) = (larg, longg) and always index sand[y, x]
sand = np.zeros((larg, longg), dtype=np.uint8)

# goal / home
x_destination, y_home = longg - 50, larg - 50
swap_goal = False

# RL brain (5 inputs, 3 actions, gamma=0.9)
brain = DQN(5, 3, 0.9)
last_reward = 0.0
last_distance = 0.0
scores = []

# mapping from action index -> rotation delta
action_rotation = {0: -20, 1: 0, 2: 20}


# ========= Utility =========
def clamp_patch(cx, cy, pad=10):
    """Return integer (x0, x1, y0, y1) bounds within the world for a square around (cx, cy)."""
    ix = int(np.clip(cx, 0, longg - 1))
    iy = int(np.clip(cy, 0, larg - 1))
    x0 = max(ix - pad, 0); x1 = min(ix + pad, longg - 1)
    y0 = max(iy - pad, 0); y1 = min(iy + pad, larg  - 1)
    return x0, x1, y0, y1


# ========= Car Actor =========
class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    sensor1_x = NumericProperty(0); sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0); sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0); sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    signal1 = NumericProperty(0.0)
    signal2 = NumericProperty(0.0)
    signal3 = NumericProperty(0.0)

    def move(self, rotation):
        # advance then rotate for next step
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = (self.angle + self.rotation) % 360

        # 3 short-range sensors
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        # sample small square patches around the sensor tips
        for idx, s in enumerate((self.sensor1, self.sensor2, self.sensor3), start=1):
            x0, x1, y0, y1 = clamp_patch(*s, pad=10)
            # NOTE: sand[y, x]
            val = float(np.sum(sand[y0:y1+1, x0:x1+1])) / ((y1 - y0 + 1) * (x1 - x0 + 1))
            setattr(self, f"signal{idx}", val)


# ========= Game Loop =========
class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(5, 0)

    def _car_on_sand(self):
        ix = int(np.clip(self.car.x, 0, longg - 1))
        iy = int(np.clip(self.car.y, 0, larg  - 1))
        return sand[iy, ix] > 0  # NOTE: sand[y, x]

    def update(self, dt):
      global last_reward, last_distance, x_destination, y_home, swap_goal

      # --- base speed & surface handling (unchanged) ---
      left_margin, bottom_margin = 10, 10
      right_margin  = 10
      top_margin    = 10

      # keep your speed logic
      if self._car_on_sand():
          self.car.velocity = Vector(2, 0).rotate(self.car.angle)
          last_reward = -3.0
      else:
          self.car.velocity = Vector(6, 0).rotate(self.car.angle)
          last_reward = +0.05

      # --- goal guidance (unchanged) ---
      distance = ((self.car.x - x_destination) ** 2 + (self.car.y - y_home) ** 2) ** 0.5
      vx, vy = self.car.velocity
      to_goal = (x_destination - self.car.x, y_home - self.car.y)
      orient_deg = Vector(vx, vy).angle(to_goal)
      orient_score = 1.0 - (orient_deg / 180.0)
      last_reward += 0.15 * orient_score

      if last_distance and distance < last_distance:
          last_reward += 0.2
      else:
          last_reward -= 0.2

      # state signal (unchanged)
      last_signal = [self.car.signal1, self.car.signal2, self.car.signal3,
                    orient_score, -orient_score]

      # learn + choose (unchanged)
      action = brain.update(last_reward, last_signal)
      rotation = action_rotation.get(action, 0)
      self.car.move(rotation)  # <-- position updated here

      # ===== BOUNDS FIX (only change) =====
      # clamp AFTER move(), using Game widget size
      left   = left_margin
      bottom = bottom_margin
      right  = self.width  - right_margin
      top    = self.height - top_margin

      clamped = False
      if self.car.x < left:
          self.car.x = left
          clamped = True
      if self.car.y < bottom:
          self.car.y = bottom
          clamped = True
      if self.car.right > right:
          self.car.right = right
          clamped = True
      if self.car.top > top:
          self.car.top = top
          clamped = True

      if clamped:
          last_reward -= 1.0
      # ===== END BOUNDS FIX =====

      # update sensor balls (unchanged)
      self.ball1.pos = self.car.sensor1
      self.ball2.pos = self.car.sensor2
      self.ball3.pos = self.car.sensor3

      # swap goal when close (unchanged)
      if distance < 25:
          swap_goal = not swap_goal
          if swap_goal:
              x_destination, y_home = 50, 50
          else:
              x_destination, y_home = self.width - 50, self.height - 50

      last_distance = distance

    # ========= Drawing (writes into sand[y, x]) =========
    def _write_sand(self, x, y, r=10):
        ix = int(np.clip(x, 0, longg - 1))
        iy = int(np.clip(y, 0, larg  - 1))
        x0 = max(ix - r, 0); x1 = min(ix + r, longg - 1)
        y0 = max(iy - r, 0); y1 = min(iy + r, larg  - 1)
        sand[y0:y1+1, x0:x1+1] = 1  # NOTE: sand[y, x]

    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 0, 1)  # yellow line you see
            touch.ud["line"] = Line(points=[touch.x, touch.y], width=12)
        self._write_sand(touch.x, touch.y)
        return True

    def on_touch_move(self, touch):
        if "line" in touch.ud:
            touch.ud["line"].points += [touch.x, touch.y]
        self._write_sand(touch.x, touch.y)
        return True


# ========= App =========
class App_(App):
    def build(self):
        game = Game(size=(longg, larg))

        # Create and attach visual widgets from KV classes
        game.car = Factory.Car()
        game.add_widget(game.car)

        game.ball1 = Factory.Ball1()
        game.ball2 = Factory.Ball2()
        game.ball3 = Factory.Ball3()
        game.add_widget(game.ball1)
        game.add_widget(game.ball2)
        game.add_widget(game.ball3)

        Clock.schedule_interval(game.update, 1.0 / 60.0)
        game.serve_car()
        return game

    def on_stop(self):
        brain.save()

    def save(self, _obj=None):
        brain.save()

    def load(self, _obj=None):
        brain.load()


if __name__ == "__main__":
    App_().run()
