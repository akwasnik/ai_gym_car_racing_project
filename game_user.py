import gymnasium as gym
import numpy as np
import pygame


env = gym.make('CarRacing-v3', render_mode='human')
observation, info = env.reset()


pygame.init()
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    steering = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steering = -1.0
    if keys[pygame.K_RIGHT]:
        steering = 1.0
    if keys[pygame.K_UP]:
        gas = 1.0
    if keys[pygame.K_DOWN]:
        brake = 0.8

    action = np.array([steering, gas, brake], dtype=np.float32)

    
    observation, reward, terminated, truncated, info = env.step(action)


    env.render()

    if terminated or truncated:
        observation, info = env.reset()
        
    clock.tick(60)
env.close()
pygame.quit()

