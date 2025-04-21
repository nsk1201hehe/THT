import pygame
import pyaudio
import numpy as np
import time
import random
from collections import defaultdict

import os
# Initialize Pygame
pygame.init()

# Get screen dimensions for full screen
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# Letter scale factors
center_letter_scale = 350
top_left_letter_scale = 150

# Initialize the screen in full screen mode
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("Bouncing Circles and Sound Visualization")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
# Circle properties
initial_num_circles = 5500
min_radius = 2
max_radius = 2
circles = []

class Circle:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.target_x = None
        self.target_y = None

    def move(self):
        if self.target_x is None or self.target_y is None:
            # Regular bouncing movement
            self.x += self.vx
            self.y += self.vy
            if self.x + self.radius > screen_width or self.x - self.radius < 0:
                self.vx *= -1
            if self.y + self.radius > screen_height or self.y - self.radius < 0:
                self.vy *= -1
        else:
            # Move towards target position
            speed = 0.05
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            self.x += dx * speed
            self.y += dy * speed

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

# Create initial circles
for _ in range(initial_num_circles):
    radius = random.randint(min_radius, max_radius)
    x = random.randint(radius, screen_width - radius)
    y = random.randint(radius, screen_height - radius)
    # Set circle color to black
    color = black
    circles.append(Circle(x, y, radius, color))

# Sound parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Silent threshold
SILENT_THRESHOLD = 50

# Parameters for automatic threshold adjustment

RMS_THRESHOLD_FACTOR = 1.5
MIN_SILENT_THRESHOLD = 50 # Defined here

# Frequency ranges for letters (adjust as needed)
start_frequency = 1400
frequency_interval = 200
frequency_ranges = {}
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i, letter in enumerate(alphabet):
    lower_bound = start_frequency + i * frequency_interval
    upper_bound = lower_bound + frequency_interval - 1
    frequency_ranges[(lower_bound, upper_bound)] = letter
    if upper_bound >= 6600:
        frequency_ranges[(lower_bound, 6600)] = letter
        break
    if i == len(alphabet) - 2 and upper_bound < 6600 - frequency_interval:
        lower_bound = start_frequency + (i + 1) * frequency_interval
        frequency_ranges[(lower_bound, 6600)] = alphabet[i+1]
        break
if len(frequency_ranges) < 26:
    last_lower_bound = start_frequency + 25 * frequency_interval
    frequency_ranges[(last_lower_bound, 6600)] = 'Z'

# Add frequency range for QR code
frequency_ranges[(400, 1399)] = 'QR'

frequency_buffer = []
buffer_size = 10
buffer_weights = np.linspace(0.1, 1, buffer_size)

# Parameters for learning threshold
VALID_RMS_HISTORY_SIZE = 100
INVALID_RMS_HISTORY_SIZE = 100
valid_rms_history = []
invalid_rms_history = []
LEARNING_RATE = 0.05  # How quickly the threshold adjusts



# Function to detect dominant frequency
def detect_frequency():
    dominant_frequency = 0
    avg_valid_rms = 80
    raw_detected_letter = None
    data = stream.read(CHUNK)
    data_length = len(data)
    expected_length = CHUNK * 2
    if data_length != expected_length:
        print(f"Warning: Read {data_length} bytes, expected {expected_length}")
        if data_length == 0:
            return None

    try:
        indata = np.frombuffer(data, dtype=np.int16)
        #print(indata)
    except Exception as e:
        print(f"Error converting buffer: {e}")
        return None

    rms = np.sqrt(np.mean(indata**2))
    global SILENT_THRESHOLD

    # Check for a sudden significant rise in RMS
    if valid_rms_history:
        avg_valid_rms = np.mean(valid_rms_history)
        if avg_valid_rms - 30 > rms or rms > avg_valid_rms + 30:
            return rms  # Return the tracked RMS and do not append to valid history

    if dominant_frequency is not None and raw_detected_letter is not None:
        if valid_rms_history:
            avg_valid_rms_for_append = np.mean(valid_rms_history)
            if avg_valid_rms - 10 <= rms <= avg_valid_rms_for_append + 10:
                valid_rms_history.append(rms)
                if len(valid_rms_history) > VALID_RMS_HISTORY_SIZE:
                    valid_rms_history.pop(0)
        else:
            valid_rms_history.append(rms)
            if len(valid_rms_history) > VALID_RMS_HISTORY_SIZE:
                valid_rms_history.pop(0)
    elif rms < SILENT_THRESHOLD: # Consider this as potential silence/background
        invalid_rms_history.append(rms)
        if len(invalid_rms_history) > INVALID_RMS_HISTORY_SIZE:
            invalid_rms_history.pop(0)

    # Adjust SILENT_THRESHOLD based on history
    if valid_rms_history and invalid_rms_history:
        avg_valid_rms = np.mean(valid_rms_history)
        max_invalid_rms = max(invalid_rms_history) if invalid_rms_history else 0

        # Try to set the threshold between the typical valid RMS and the max invalid RMS
        new_threshold = max(MIN_SILENT_THRESHOLD, (avg_valid_rms + max_invalid_rms) / 2)
        SILENT_THRESHOLD = (1 - LEARNING_RATE) * SILENT_THRESHOLD + LEARNING_RATE * new_threshold
        
    elif valid_rms_history:
        SILENT_THRESHOLD = max(MIN_SILENT_THRESHOLD, np.mean(valid_rms_history) * 0.75) # Adjust factor as needed
    elif invalid_rms_history and not valid_rms_history:
        SILENT_THRESHOLD = max(MIN_SILENT_THRESHOLD, max(invalid_rms_history) * 1.2) # Adjust factor as needed

    #print(SILENT_THRESHOLD)
    if rms < SILENT_THRESHOLD  or avg_valid_rms - 20 <= rms <= avg_valid_rms + 20:
        return None
    

    n = len(indata)
    
    if n == 0:
        return None
    
    fft_data = np.fft.fft(indata)
    frequencies = np.fft.fftfreq(n, 1.0/RATE)
    positive_indices = np.where(frequencies > 0)[0]
    positive_frequencies = frequencies[positive_indices]
    magnitudes = np.abs(fft_data[positive_indices])

    dominant_frequency = None
    num_positive_frequencies = len(positive_frequencies)

    if magnitudes.any() and num_positive_frequencies > 0:
        valid_indices_mask = (positive_frequencies >= 400) & (positive_frequencies <= 6600) # Include QR code frequency
        valid_indices_relative = np.where(valid_indices_mask)[0]

        if valid_indices_relative.any():
            valid_magnitudes = magnitudes[valid_indices_relative]
            dominant_frequency_index_in_valid_relative = np.argmax(valid_magnitudes)

            if dominant_frequency_index_in_valid_relative < len(valid_indices_relative):
                original_index_in_positive = positive_indices[valid_indices_relative[dominant_frequency_index_in_valid_relative]]

                if original_index_in_positive < num_positive_frequencies:
                    dominant_frequency = int(positive_frequencies[original_index_in_positive])

    if dominant_frequency is not None:
        frequency_buffer.append(dominant_frequency)
        if len(frequency_buffer) > buffer_size:
            frequency_buffer.pop(0)

        if frequency_buffer:
            weighted_average = np.average(frequency_buffer, weights=buffer_weights[-len(frequency_buffer):])
            return int(weighted_average)
        else:
            return None
    else:
        return None
    
# Predefined positions for letters
def generate_line_points(start, end, num_points):
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0.5
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        points.append((x, y))
    return points

def generate_arc_points(center, radius, start_angle, end_angle, num_points):
    points = []
    for i in range(num_points):
        angle = start_angle + i / (num_points - 1) * (end_angle - start_angle)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append((x, y))
    return points

num_letter_circles = initial_num_circles // 15

letter_patterns = {
    'A': generate_line_points((20, 90), (50, 10), num_letter_circles) +
         generate_line_points((80, 90), (50, 10), num_letter_circles) +
         generate_line_points((30, 50), (70, 50), num_letter_circles),
    'B': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_arc_points((50, 30), 30, np.pi/2, -np.pi/2, num_letter_circles // 2) +
         generate_arc_points((50, 70), 30, np.pi/2, -np.pi/2, num_letter_circles // 2),
    'C': generate_arc_points((50, 50), 40, np.pi * 0.25, np.pi * 1.75, int(num_letter_circles * 0.8)),
    'D': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_arc_points((60, 50), 40, np.pi/2, -np.pi/2, num_letter_circles),
    'E': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_line_points((20, 10), (80, 10), num_letter_circles // 2) +
         generate_line_points((20, 50), (70, 50), num_letter_circles // 2) +
         generate_line_points((20, 90), (80, 90), num_letter_circles // 2),
    'F': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_line_points((20, 10), (80, 10), num_letter_circles // 2) +
         generate_line_points((20, 50), (70, 50), num_letter_circles // 2),
    'G': generate_arc_points((50, 50), 40, np.pi * 1.75, np.pi * 0.15, int(num_letter_circles * 0.6)) +
         generate_line_points((50, 50), (85, 50), num_letter_circles // 3) +
         generate_line_points((85, 50), (85, 70), num_letter_circles // 3),
    'H': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_line_points((80, 10), (80, 90), num_letter_circles) +
         generate_line_points((20, 50), (80, 50), num_letter_circles),
    'I': generate_line_points((50, 10), (50, 90), num_letter_circles) +
         generate_line_points((20, 10), (80, 10), num_letter_circles // 2) +
         generate_line_points((20, 90), (80, 90), num_letter_circles // 2),
    'J': generate_line_points((30, 10), (30, 80), num_letter_circles) +
         generate_arc_points((10, 80), 20, np.pi, 0, num_letter_circles // 2),
    'K': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_line_points((20, 50), (80, 10), num_letter_circles // 2) +
         generate_line_points((20, 50), (80, 90), num_letter_circles // 2),
    'L': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_line_points((20, 90), (80, 90), num_letter_circles // 2),
    'M': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_line_points((80, 10), (80, 90), num_letter_circles) +
         generate_line_points((20, 10), (50, 50), num_letter_circles // 3) +
         generate_line_points((50, 50), (80, 10), num_letter_circles // 3),
    'N': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_line_points((80, 10), (80, 90), num_letter_circles) +
         generate_line_points((20, 10), (80, 90), num_letter_circles),
    'O': generate_arc_points((50, 50), 40, 0, 2 * np.pi, int(num_letter_circles * 0.8)),
    'P': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_arc_points((50, 30), 30, np.pi/2, -np.pi/2, num_letter_circles // 2),
    'Q': generate_arc_points((50, 50), 40, 0, 2 * np.pi, int(num_letter_circles * 0.8)) +
         generate_line_points((70, 70), (90, 90), num_letter_circles // 4),
    'R': generate_line_points((20, 10), (20, 90), num_letter_circles) +
         generate_arc_points((50, 30), 30, np.pi/2, -np.pi/2, num_letter_circles // 2) +
         generate_line_points((20, 50), (80, 90), num_letter_circles // 2),
    'S': generate_arc_points((54, 40), 20, np.pi, 0, num_letter_circles // 2) +
         generate_arc_points((86, 60), 20, -np.pi, np.pi * 0., num_letter_circles // 2),
    'T': generate_line_points((50, 10), (50, 90), num_letter_circles) +
         generate_line_points((20, 10), (80, 10), num_letter_circles // 2),
    'U': generate_line_points((20, 10), (20, 70), num_letter_circles) +
         generate_line_points((80, 10), (80, 70), num_letter_circles) +
         generate_arc_points((50, 70), 30, np.pi, 0, num_letter_circles // 2),
    'V': generate_line_points((20, 10), (50, 90), num_letter_circles) +
         generate_line_points((50, 90), (80, 10), num_letter_circles),
    'W': generate_line_points((20, 10), (40, 90), num_letter_circles // 2) +
         generate_line_points((40, 90), (50, 30), num_letter_circles // 4) +
         generate_line_points((50, 30), (60, 90), num_letter_circles // 4) +
         generate_line_points((60, 90), (80, 10), num_letter_circles // 2),
    'X': generate_line_points((20, 10), (80, 90), num_letter_circles) +
         generate_line_points((80, 10), (20, 90), num_letter_circles),
    'Y': generate_line_points((50, 50), (50, 120), num_letter_circles) +
         generate_line_points((20, 10), (50, 60), num_letter_circles // 2) +
         generate_line_points((80, 10), (50, 60), num_letter_circles // 2),
    'Z': generate_line_points((20, 10), (80, 10), num_letter_circles // 2) +
         generate_line_points((80, 10), (20, 90), num_letter_circles) +
         generate_line_points((20, 90), (80, 90), num_letter_circles // 2),
}

def get_scaled_letter_positions(letter, scale, offset_x, offset_y):
    if letter in letter_patterns:
        pattern = letter_patterns[letter]
        if not pattern:
            return []
        min_x = min(p[0] for p in pattern)
        max_x = max(p[0] for p in pattern)
        min_y = min(p[1] for p in pattern)
        max_y = max(p[1] for p in pattern)

        width = max_x - min_x
        height = max_y - min_y
        scale_factor_width = scale / width if width > 0 else 1
        scale_factor_height = scale / height if height > 0 else 1
        scale_factor = min(scale_factor_width, scale_factor_height)

        scaled_pattern = [(int((p[0] - min_x) * scale_factor), int((p[1] - min_y) * scale_factor)) for p in pattern]

        letter_width = int((max(sp[0] for sp in scaled_pattern) - min(sp[0] for sp in scaled_pattern)))
        letter_height = int((max(sp[1] for sp in scaled_pattern) - min(sp[1] for sp in scaled_pattern)))

        final_positions = [(sp[0] + offset_x, sp[1] + offset_y) for sp in scaled_pattern]
        return final_positions
    return None

def get_scaled_qr_code_positions(scale, offset_x, offset_y, density_factor=3):
    """
    Generates positions for circles forming a dense QR code.

    Args:
        scale (int): The overall scale of the QR code.
        offset_x (int): The horizontal offset for the QR code center.
        offset_y (int): The vertical offset for the QR code center.
        density_factor (int): The number of smaller circles to fit along one side of a QR module.
    """
    # This is a placeholder QR code pattern (21x21)
    qr_pattern = [
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
]
    qr_size = len(qr_pattern)
    circle_positions = []
    module_scale = scale / qr_size
    module_offset_x = offset_x - (qr_size * module_scale) // 2
    module_offset_y = offset_y - (qr_size * module_scale) // 2

    for row in range(qr_size):
        for col in range(qr_size):
            if qr_pattern[row][col] == 1: # Assuming 1 represents a black block
                # Calculate the top-left corner of the module
                module_x = int(col * module_scale + module_offset_x)
                module_y = int(row * module_scale + module_offset_y)

                # Calculate spacing for smaller circles within the module
                circle_diameter = module_scale // density_factor
                circle_radius = circle_diameter // 2
                inner_offset = module_scale // (density_factor + 1) if density_factor > 1 else module_scale // 2

                for i in range(density_factor):
                    for j in range(density_factor):
                        # Calculate the center of each small circle within the module
                        center_x = module_x + inner_offset + i * (module_scale // density_factor)
                        center_y = module_y + inner_offset + j * (module_scale // density_factor)

                        circle_positions.append((int(center_x), int(center_y)))

    return circle_positions

# State management
BOUNCING = 0
CENTER_FORMING = 1
CENTER_HOLDING = 2
CENTER_DISSOLVING = 3

center_state = BOUNCING
detected_letter = None
stable_letter = None
letter_stable_counter = 0
letter_stability_threshold = 1

center_letter = None
center_form_start_time = 0
center_hold_duration = 10
center_hold_start_time = 0
center_letter_circles = []  # To track circles forming the center letter

# Top Left Letter Management
class TopLeftLetter:
    def __init__(self, letter, scale, start_x, start_y):
        self.letter = letter
        self.scale = scale
        self.start_x = start_x
        self.start_y = start_y
        self.form_start_time = time.time()
        self.hold_duration = 3
        self.hold_start_time = 0
        self.dissolve_start_time = 0
        self.positions = get_scaled_letter_positions(letter, scale, start_x, start_y)
        self.state = "FORMING" # FORMING, HOLDING, DISSOLVING
        self.arranged_circles = [] # Keep track of circles used for this letter
        if self.positions:
            min_x = min(p[0] for p in self.positions)
            max_x = max(p[0] for p in self.positions)
            self.width = max_x - min_x
        else:
            self.width = 0

active_top_left_letters = []
top_left_start_x = 10
top_left_start_y = 10
top_left_spacing = 10
next_top_left_x = top_left_start_x

def arrange_circles_to_positions(positions, assigned_circles):
    if not positions:
        return

    available_circles = [c for c in circles if c not in assigned_circles and c.target_x is None]
    random.shuffle(available_circles)

    num_circles_to_arrange = min(len(positions), len(available_circles))
    for i in range(num_circles_to_arrange):
        available_circles[i].target_x = positions[i][0]
        available_circles[i].target_y = positions[i][1]
        available_circles[i].vx = 0
        available_circles[i].vy = 0
        assigned_circles.append(available_circles[i])
        if assigned_circles is center_letter_circles: # Specifically track center letter circles
            pass # Already appending to the correct list

def reset_circle_targets(letter_circles=None):
    if letter_circles is None:
        for circle in circles:
            circle.target_x = None
            circle.target_y = None
            circle.vx = random.uniform(-2, 2)
            circle.vy = random.uniform(-2, 2)
    else:
        for circle in letter_circles:
            circle.target_x = None
            circle.target_y = None
            circle.vx = random.uniform(-2, 2)
            circle.vy = random.uniform(-2, 2)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Set screen color to white
    screen.fill(white)

    current_frequency = detect_frequency()
    raw_detected_letter = None

    if current_frequency is not None:
        for freq_range, letter in frequency_ranges.items():
            if freq_range[0] <= current_frequency <= freq_range[1]:
                raw_detected_letter = letter
                break

    if raw_detected_letter == stable_letter:
        letter_stable_counter += 1
        if letter_stable_counter >= letter_stability_threshold:
            detected_letter = stable_letter
    else:
        stable_letter = raw_detected_letter
        letter_stable_counter = 0
        detected_letter = None

    # Center Letter Formation Logic
    if center_state != CENTER_FORMING and center_state != CENTER_HOLDING and detected_letter:
        center_letter = detected_letter
        center_state = CENTER_FORMING
        center_form_start_time = time.time()
        center_letter_circles = [] # Clear the list for a new center display
        if center_letter == 'QR':
            qr_positions = get_scaled_qr_code_positions(center_letter_scale, screen_width // 2, screen_height // 2)
            arrange_circles_to_positions(qr_positions, center_letter_circles)
        else:
            center_positions = get_scaled_letter_positions(center_letter, center_letter_scale, screen_width // 2, screen_height // 2)
            if center_positions:
                min_x = min(p[0] for p in center_positions)
                max_x = max(p[0] for p in center_positions)
                min_y = min(p[1] for p in center_positions)
                max_y = max(p[1] for p in center_positions)
                center_offset_x = screen_width // 2 - (max_x + min_x) // 2
                center_offset_y = screen_height // 2 - (max_y + min_y) // 2
                final_center_positions = [(p[0] + center_offset_x, p[1] + center_offset_y) for p in center_positions]
                arrange_circles_to_positions(final_center_positions, center_letter_circles) # Track center letter circles

    elif center_state == CENTER_FORMING:
        if time.time() - center_form_start_time > 1: # Small delay to let formation begin
            center_state = CENTER_HOLDING
            center_hold_start_time = time.time()

    elif center_state == CENTER_HOLDING:
        if time.time() - center_hold_start_time >= center_hold_duration:
            center_state = CENTER_DISSOLVING
            reset_circle_targets(center_letter_circles) # Reset targets for center letter circles only
            # Create a new top-left letter
            new_top_left = TopLeftLetter(center_letter, top_left_letter_scale, next_top_left_x, top_left_start_y)
            active_top_left_letters.append(new_top_left)
            arrange_circles_to_positions(new_top_left.positions, new_top_left.arranged_circles)
            next_top_left_x += new_top_left.width + top_left_spacing
            if next_top_left_x > screen_width - 50:
                next_top_left_x = top_left_start_x

    elif center_state == CENTER_DISSOLVING:
        if all(c.target_x is None for c in center_letter_circles): # Wait for center letter to dissolve
            center_state = BOUNCING

    # Top Left Letter Management
    for tl_letter in list(active_top_left_letters):
        # Debugging information
        # print(f"Top Left Letter: {tl_letter.letter}, State: {tl_letter.state}, Hold Start: {tl_letter.hold_start_time}, Current Time: {time.time()}, Elapsed Hold Time: {time.time() - tl_letter.hold_start_time}")
        if tl_letter.state == "FORMING":
            if time.time() - tl_letter.form_start_time > 1:
                tl_letter.state = "HOLDING"
                tl_letter.hold_start_time = time.time()
        elif tl_letter.state == "HOLDING":
            if time.time() - tl_letter.hold_start_time >= tl_letter.hold_duration:
                tl_letter.state = "DISSOLVING"
                tl_letter.dissolve_start_time = time.time()
                reset_circle_targets(tl_letter.arranged_circles)
        elif tl_letter.state == "DISSOLVING":
            if all(c.target_x is None for c in tl_letter.arranged_circles):
                active_top_left_letters.remove(tl_letter)
                # print(f"Removed Top Left Letter: {tl_letter.letter}") # Debugging

    

    for circle in circles:
        circle.move()
        circle.draw(screen)

    

    pygame.display.flip()
    time.sleep(0.01)

# Close audio stream
stream.stop_stream()
stream.close()
p.terminate()

# Quit Pygame
pygame.quit()
os.system('cls' if os.name == 'nt' else "clear")