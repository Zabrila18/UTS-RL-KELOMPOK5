import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
x_values = []
y_values = []

pygame.init()
print("starting")

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Two actions: left or right
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(x)

class QLearningAgent:
    def __init__(self, learning_rate=0.00005, discount_factor=0.99, exploration_prob=1.0, exploration_decay=0.998):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

    def select_action(self, state):
        if random.random() < self.exploration_prob:
            return random.randint(0, 1) # Random action
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
            return torch.argmax(q_values).item() # Greedy action

    def train(self, state, action, reward, next_state, done):
        with torch.no_grad():
            target_q_values = self.model(torch.FloatTensor(next_state))
            max_target_q_value = torch.max(target_q_values)

        q_values = self.model(torch.FloatTensor(state))
        target_q_value = reward + (1 - done) * self.discount_factor * max_target_q_value
        target_q_values = q_values.clone().detach()
        target_q_values[action] = target_q_value

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.exploration_prob *= self.exploration_decay

agent = QLearningAgent()

def play(agent):
    WIDTH, HEIGHT = 500, 450
    BALL_RADIUS = 10
    PADDLE_WIDTH, PADDLE_HEIGHT = 80, 10
    BRICK_WIDTH, BRICK_HEIGHT = 40, 20
    BRICK_ROWS, BRICK_COLS = 5, 10
    WHITE = (255, 255, 255)

    total_reward = 0

    def create_bricks():
        bricks = []
        for i in range(BRICK_ROWS):
            for j in range(BRICK_COLS):
                brick = pygame.Rect(j * (BRICK_WIDTH + 5) + 25, i * (BRICK_HEIGHT + 5) + 25, BRICK_WIDTH, BRICK_HEIGHT)
                bricks.append(brick)
        return bricks

    paddle = pygame.Rect(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - 30, PADDLE_WIDTH, PADDLE_HEIGHT)
    
    lowest_brick_point = (BRICK_HEIGHT + 5) * BRICK_ROWS + 25
#    ball_x = random.randint(BALL_RADIUS, WIDTH - BALL_RADIUS)
#    ball_y = random.randint(lowest_brick_point + BALL_RADIUS, HEIGHT // 2)
    ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)

    bricks = create_bricks()

    ball_dx = random.choice([3, -3])
    ball_dy = 3

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    def get_state_vector(ball, paddle, ball_dx, ball_dy, bricks):
        state_vector = [0] * 8
        
        # Ball's position relative to the paddle
        if ball.centerx < paddle.centerx:
            state_vector[0] = 1
        else:
            state_vector[1] = 1

        # Ball's direction
        if ball_dy < 0 and ball_dx > 0:
            state_vector[2] = 1 # North East
        elif ball_dy < 0 and ball_dx < 0:
            state_vector[3] = 1 # North West
        elif ball_dy > 0 and ball_dx > 0:
            state_vector[4] = 1 # South East
        elif ball_dy > 0 and ball_dx < 0:
            state_vector[5] = 1 # South West

        left_bricks_present = any(brick.centerx < paddle.centerx for brick in bricks)
        right_bricks_present = any(brick.centerx > paddle.centerx for brick in bricks)

        if left_bricks_present:
            state_vector[6] = 1
        if right_bricks_present:
            state_vector[7] = 1

        return state_vector

    # Inside the game loop
    state_vector = get_state_vector(ball, paddle, ball_dx, ball_dy, bricks)

    running = True
    while running:
        reward = 0
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if len(bricks) == 0:
            running = False
        total_reward
            
        state_vector = get_state_vector(ball, paddle, ball_dx, ball_dy, bricks)
        action = agent.select_action(state_vector)
        if action == 0 and paddle.left > 0:
            paddle.move_ip(-5, 0)
        elif action != 0 and paddle.right < WIDTH:
            paddle.move_ip(5, 0)

        ball.move_ip(ball_dx, ball_dy)
        if ball.left <= 0 or ball.right >= WIDTH:
            ball_dx = -ball_dx
        if ball.top <= 0:
            ball_dy = -ball_dy
        if ball.bottom >= HEIGHT:
            distance_from_paddle = abs((paddle.left + paddle.right) / 2 - (ball.left + ball.right) / 2)
            reward = 0 - distance_from_paddle / 25
            running = False
        if ball.colliderect(paddle):
            if ball_dy < 0:
                ball_dy = random.uniform(2, 4)
            else:
                ball_dy = 0 - random.uniform(2, 4)
            reward = 15
        for brick in bricks[:]:
            if ball.colliderect(brick):
                reward = 2
                bricks.remove(brick)
                ball_dy = -ball_dy

        pygame.draw.ellipse(screen, (0, 0, 0), ball)
        pygame.draw.rect(screen, (0, 0, 0), paddle)
        for brick in bricks:
            pygame.draw.rect(screen, (0, 0, 0), brick)

        pygame.display.flip()
        next_state_vector = get_state_vector(ball, paddle, ball_dx, ball_dy, bricks)
        agent.train(state_vector, action, reward, next_state_vector, running)
        total_reward += reward

        if not running:
            return total_reward

        clock.tick(60)
def plot_xy(x_values, y_values, title="Progress", x_label="Iteration", y_label="Score"):
    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Display the plot
    plt.show()
    
for i in range(200):
    reward = int(play(agent))
    x_values.append(i)
    y_values.append(reward)
    print(i, ". ", reward, sep = "")

plot_xy(x_values, y_values)
