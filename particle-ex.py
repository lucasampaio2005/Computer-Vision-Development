import random 
import time
import cv2
import numpy as np

frame_height, frame_width = 600, 600

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('simulation_detection.mp4', fourcc, 50.0, (frame_width, frame_height))

def create_particle():
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    radius = 10
    uniform_random = np.random.uniform()

    if uniform_random <= 0.25:
        position = (random.randint(radius, frame_width - radius), radius)
        angle = random.randint(0, 180)
        start_pos = "bottom"
    elif uniform_random <= 0.5:
        position = (random.randint(radius, frame_width - radius), frame_height - radius)
        angle = random.randint(180, 360)
        start_pos  ="top"
    elif uniform_random <= 0.75:
        position = (radius, random.randint(radius, frame_height - radius))
        angle = random.randint(-90, 90)
        start_pos = "left"
    else:
        position = (frame_width - radius, random.randint(radius, frame_height - radius))
        angle = random.randint(90, 270)
        start_pos  = "right"
    
    return {'position' : position, 'color' : color, 'radius' : radius, 'angle' : angle, 'start_pos' : start_pos}

def move_particle(particle):
    if particle['start_pos'] == 'bottom':
        angle = random.randint(0, 180)
    elif particle['start_pos'] == 'top':
        angle = random.randint(180, 360)
    elif particle['start_pos'] == 'left':
        angle = random.randint(-90, 90)
    elif particle['start_pos'] == 'right':
        angle = random.randint(90, 270)

    angle_rad = np.deg2rad(angle)
    dx = int(particle['radius'] * np.cos(angle_rad))
    dy = int(particle['radius'] * np.sin(angle_rad))
    x, y = particle['position']
    particle['position'] = (x + dx, y + dy)

def is_off_screen(particle):
    x, y = particle['position']
    return x < 1 or x > frame_width-1 or y < 1 or y > frame_height-1

def draw_frame(particles):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    bounding_boxes = []
    for particle in particles:
        cv2.circle(frame, particle['position'], particle['radius'], particle['color'], -1)
        x, y = particle['position']
        # cv2.rectangle(frame, (x - 2* particle['radius'], y - 2 * particle['radius']), (x + 2 * particle['radius'], y + 2 * particle['radius']), (0, 255, 0), 1)
        bounding_boxes.append({'x_center': x, 'y_center': y, 'width': particle['radius'], 'height': particle['radius']})
        
    return frame, bounding_boxes

def simulate_particles(total_data):
    particles = []
    max_particles = 50
    total_particles_created = 0
    timer = 0 

    while len(particles) > 0 or total_particles_created < max_particles:
        if total_particles_created < max_particles and timer % 5 == 0:
            total_particles_created += 1
            particles.append(create_particle())

        for particle in particles[:]:
            move_particle(particle)
            if is_off_screen(particle):
                particles.remove(particle)

        frame, bounding_boxes = draw_frame(particles)
        total_data.append({'frame': frame, 'boundary_boxes': bounding_boxes})
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        timer += 1

        out.release()
        cv2.destroyAllWindows()

        return total_data
    
total_data = []
for i in range(12):
    total_data = simulate_particles(total_data)