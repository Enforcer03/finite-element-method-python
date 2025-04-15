import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

from scipy.special import factorial

def abs_polynomial_approx(x, n):
    """
    Calculate the polynomial approximation of |x| using the given formula:
    p_n(x) = sum_{k=0}^n ((-1)^k(2k)!)/(2^(2k)(k!)^2(2k+1)) * x^(2k+1)
    """
    result = np.zeros_like(x)
    for k in range(n+1):
        # Calculate coefficient
        coef = ((-1)**k * factorial(2*k)) / (2**(2*k) * (factorial(k)**2) * (2*k+1))
        # Add term to result
        result += coef * x**(2*k+1)
    return result

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
plt.title('Polynomial Approximation of |x|')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Generate x values
x = np.linspace(-1, 1, 1000)
# True absolute value function
y_true = np.abs(x)

# Plot the true function
true_line, = plt.plot(x, y_true, 'k-', label='|x|')

# Initialize the approximation line
approx_line, = plt.plot([], [], 'r-', label='Approximation')

# Text for displaying current n value
n_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Set up legend
plt.legend()

# Set fixed y-axis limits to prevent rescaling during animation
plt.ylim(-0.1, 1.1)

def init():
    approx_line.set_data([], [])
    n_text.set_text('')
    return approx_line, n_text

def update(frame):
    n = frame + 1  # Start from n=1
    y_approx = abs_polynomial_approx(x, n)
    approx_line.set_data(x, y_approx)
    n_text.set_text(f'n = {n}')
    return approx_line, n_text

# Create animation
max_n = 30
ani = FuncAnimation(fig, update, frames=max_n, init_func=init, blit=True, interval=200)

# Instead of directly saving as GIF, try this approach:
try:
    # Try using pillow writer
    ani.save('abs_polynomial_approximation.gif', writer='pillow', fps=2)
except Exception as e:
    print(f"Error with pillow writer: {e}")
    try:
        # Try using imagemagick if available
        ani.save('abs_polynomial_approximation.gif', writer='imagemagick', fps=2)
    except Exception as e:
        print(f"Error with imagemagick: {e}")
        print("Saving individual frames instead...")
        
        # Save individual frames as a workaround
        for i in range(max_n):
            update(i)
            plt.savefig(f'abs_approx_frame_{i+1:03d}.png')
        
        print("Done saving frames. To create a GIF manually, you can use:")
        print("ImageMagick: convert -delay 50 abs_approx_frame_*.png abs_polynomial_approximation.gif")
        print("or FFmpeg: ffmpeg -framerate 2 -i abs_approx_frame_%03d.png abs_polynomial_approximation.gif")

plt.close()

# Create another visualization showing multiple approximations at once
plt.figure(figsize=(12, 8))
plt.title('Polynomial Approximations of |x| for Different n')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Plot true function
plt.plot(x, y_true, 'k-', linewidth=2, label='|x|')

# Plot various approximations with a color gradient
n_values = [1, 3, 5, 10, 20, 30]
colors = cm.viridis(np.linspace(0, 1, len(n_values)))

for i, n in enumerate(n_values):
    y_approx = abs_polynomial_approx(x, n)
    plt.plot(x, y_approx, '-', color=colors[i], linewidth=1.5, label=f'n = {n}')

plt.legend()
plt.savefig('abs_polynomial_approximations_comparison.png', dpi=300)
plt.show()

# Calculate and plot the error for different n values
plt.figure(figsize=(10, 6))
plt.title('L² Error in Polynomial Approximation of |x|')
plt.xlabel('n')
plt.ylabel('Error')
plt.grid(True)

errors = []
n_range = range(1, 41)

for n in n_range:
    y_approx = abs_polynomial_approx(x, n)
    # Calculate L² error (approximated by the sum of squared differences)
    error = np.sqrt(np.mean((y_approx - y_true)**2))
    errors.append(error)

plt.plot(n_range, errors, 'bo-')
plt.yscale('log')  # Use log scale to see the error decay better
plt.savefig('abs_polynomial_approximation_error.png', dpi=300)
plt.show()