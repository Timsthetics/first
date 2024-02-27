from numpy import *

def compute_error_for_line_at_given_point(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b))**2
    return totalError / float(len(points))

def gradient_decent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += -(2/n) * (y - ((m_current * x) + b_current))
        m_gradient += (2/n) * x * (y - (m_current * x) + b_current)

    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]

def run():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    initial_error = compute_error_for_line_at_given_point(initial_b, initial_m, points)
    print(f'Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {initial_error}')

    [b, m] = gradient_decent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    final_error = compute_error_for_line_at_given_point(b, m, points)
    print(f'Ending gradient descent at b = {b}, m = {m}, error = {final_error}')

if __name__ == '__main__':
    run()
