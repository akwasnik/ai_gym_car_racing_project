import gymnasium as gym
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# === 1. Define fuzzy variables ===
angle = ctrl.Antecedent(np.linspace(-np.pi, np.pi, 101), 'angle')
speed = ctrl.Antecedent(np.linspace(0, 120, 121), 'speed')
steering = ctrl.Consequent(np.linspace(-1, 1, 101), 'steering')
throttle = ctrl.Consequent(np.linspace(0, 1, 101), 'throttle')

# === 2. Membership functions ===
angle['neg_large']  = fuzz.trapmf(angle.universe, [-np.pi, -np.pi, -1.5, -0.5])
angle['neg_small']  = fuzz.trimf(angle.universe, [-1.0, -0.5,  0.0])
angle['zero']       = fuzz.trimf(angle.universe, [-0.1,  0.0,  0.1])
angle['pos_small']  = fuzz.trimf(angle.universe, [ 0.0,  0.5,  1.0])
angle['pos_large']  = fuzz.trapmf(angle.universe, [ 0.5,  1.5,  np.pi,  np.pi])

speed['low']    = fuzz.trapmf(speed.universe, [0, 0, 20, 40])
speed['medium'] = fuzz.trimf(speed.universe, [30, 60, 90])
speed['high']   = fuzz.trapmf(speed.universe, [80, 100, 120, 120])

steering['left_hard']   = fuzz.trapmf(steering.universe, [-1.0, -1.0, -0.6, -0.3])
steering['left_soft']   = fuzz.trimf(steering.universe, [-0.5, -0.25, 0.0])
steering['straight']    = fuzz.trimf(steering.universe, [-0.1, 0.0, 0.1])
steering['right_soft']  = fuzz.trimf(steering.universe, [0.0, 0.25, 0.5])
steering['right_hard']  = fuzz.trapmf(steering.universe, [0.3, 0.6, 1.0, 1.0])

throttle['brake']       = fuzz.trapmf(throttle.universe, [0.0, 0.0, 0.1, 0.3])
throttle['coast']       = fuzz.trimf(throttle.universe, [0.2, 0.5, 0.8])
throttle['accelerate'] = fuzz.trapmf(throttle.universe, [0.7, 0.9, 1.0, 1.0])

# === 3. Separate rule lists ===
steering_rules = []
steering_rules.append(ctrl.Rule(angle['neg_large'], steering['right_hard']))
steering_rules.append(ctrl.Rule(angle['neg_small'], steering['right_soft']))
steering_rules.append(ctrl.Rule(angle['zero'],      steering['straight']))
steering_rules.append(ctrl.Rule(angle['pos_small'], steering['left_soft']))
steering_rules.append(ctrl.Rule(angle['pos_large'], steering['left_hard']))

throttle_rules = []
throttle_rules.append(ctrl.Rule(speed['low'],    throttle['accelerate']))
throttle_rules.append(ctrl.Rule(speed['medium'], throttle['coast']))
throttle_rules.append(ctrl.Rule(speed['high'],   throttle['brake']))

# === 4. Build control systems ===
steer_ctrl    = ctrl.ControlSystem(steering_rules)
steer_sim     = ctrl.ControlSystemSimulation(steer_ctrl)

throttle_ctrl = ctrl.ControlSystem(throttle_rules)
throttle_sim  = ctrl.ControlSystemSimulation(throttle_ctrl)

# === 5. Run controller ===
if __name__ == '__main__':
    env = gym.make('CarRacing-v3', render_mode='human')
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        unwrapped = env.unwrapped
        angle_err = unwrapped.car.hull.angle
        vel = np.hypot(unwrapped.car.hull.linearVelocity.x,
                       unwrapped.car.hull.linearVelocity.y) * 3.6

        # Fuzzy compute steering
        steer_sim.input['angle'] = angle_err
        steer_sim.compute()
        steer = float(steer_sim.output['steering'])

        # Fuzzy compute throttle
        throttle_sim.input['speed'] = vel
        throttle_sim.compute()
        accel = float(throttle_sim.output['throttle'])

        brake = 0.0
        if accel < 0.2:
            brake = (0.2 - accel) / 0.2

        action = np.array([steer, accel, brake], dtype=np.float32)
        obs, rew, done, truncated, info = env.step(action)
        total_reward += rew

    print(f"Episode finished. Total reward: {total_reward:.2f}")
    env.close()
