import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

def abs_function(x):
    """The absolute value function |x|"""
    return np.abs(x)

def monomial_basis(x, k):
    """Monomial basis function: x^k"""
    return x**k

def legendre_basis(x, k):
    """Legendre polynomial of degree k"""
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x
    else:
        # Use recurrence relation for Legendre polynomials
        P_km1 = legendre_basis(x, k-1)
        P_km2 = legendre_basis(x, k-2)
        return ((2*k-1)*x*P_km1 - (k-1)*P_km2)/k

def chebyshev_basis(x, k):
    """Chebyshev polynomial of first kind of degree k"""
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x
    else:
        # Use recurrence relation for Chebyshev polynomials
        T_km1 = chebyshev_basis(x, k-1)
        T_km2 = chebyshev_basis(x, k-2)
        return 2*x*T_km1 - T_km2

def compute_l2_projection(basis_func, max_degree, weight_func=None):
    """
    Compute L2 projection coefficients for |x| using the specified basis
    
    Parameters:
    -----------
    basis_func : function
        The basis function to use
    max_degree : int
        Maximum degree for the projection
    weight_func : function, optional
        Weight function for the inner product
        
    Returns:
    --------
    coeffs : array
        Projection coefficients
    """
    a, b = -1, 1  # Integration limits
    
    # Initialize mass matrix and load vector
    M = np.zeros((max_degree + 1, max_degree + 1))
    load_vector = np.zeros(max_degree + 1)
    
    # Compute mass matrix entries
    for i in range(max_degree + 1):
        for j in range(max_degree + 1):
            # Define integrand for mass matrix: phi_i * phi_j * weight
            def integrand_M(x):
                phi_i = basis_func(np.array([x]), i)
                phi_j = basis_func(np.array([x]), j)
                weight = 1.0 if weight_func is None else weight_func(x)
                return phi_i * phi_j * weight
            
            # Compute the integral
            M[i, j], _ = quad(integrand_M, a, b)
    
    # Compute load vector entries
    for i in range(max_degree + 1):
        # Define integrand for load vector: f * phi_i * weight
        def integrand_b(x):
            phi_i = basis_func(np.array([x]), i)
            weight = 1.0 if weight_func is None else weight_func(x)
            return abs_function(x) * phi_i * weight
        
        # Compute the integral
        load_vector[i], _ = quad(integrand_b, a, b)
    
    # Solve the linear system
    try:
        coeffs = np.linalg.solve(M, load_vector)
    except np.linalg.LinAlgError:
        # If matrix is singular or poorly conditioned, use least squares solution
        print(f"Warning: Matrix is poorly conditioned or singular. Using least squares for {max_degree} basis functions.")
        coeffs, residuals, rank, s = np.linalg.lstsq(M, load_vector, rcond=None)
    
    return coeffs

def evaluate_l2_projection(x, coeffs, basis_func):
    """
    Evaluate L2 projection at points x using the given coefficients and basis
    
    Parameters:
    -----------
    x : array-like
        Points at which to evaluate the projection
    coeffs : list
        Projection coefficients
    basis_func : function
        The basis function used for projection
        
    Returns:
    --------
    y : array
        Projected function values at x
    """
    y = np.zeros_like(x, dtype=float)
    
    for k, c_k in enumerate(coeffs):
        y += c_k * basis_func(x, k)
    
    return y

# Chebyshev weight function
def chebyshev_weight(x):
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    return 1.0 / np.sqrt(1 - x**2 + epsilon)

# Create frames directory if it doesn't exist
os.makedirs("l2_projection_frames", exist_ok=True)

# Generate evaluation points
x_eval = np.linspace(-1, 1, 1000)

# True absolute value function
y_true = np.abs(x_eval)

# Debug function to check basis functions
def debug_basis_functions():
    plt.figure(figsize=(12, 8))
    plt.title('Basis Functions Comparison')
    
    # Check first few basis functions of each type
    for i in range(4):
        plt.subplot(3, 4, i+1)
        plt.title(f'Monomial k={i}')
        plt.plot(x_eval, monomial_basis(x_eval, i))
        plt.grid(True)
        
        plt.subplot(3, 4, i+5)
        plt.title(f'Legendre k={i}')
        plt.plot(x_eval, legendre_basis(x_eval, i))
        plt.grid(True)
        
        plt.subplot(3, 4, i+9)
        plt.title(f'Chebyshev k={i}')
        plt.plot(x_eval, chebyshev_basis(x_eval, i))
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("basis_functions_debug.png")
    plt.close()

# Run debug function
debug_basis_functions()

# Define basis functions to compare
basis_functions = [
    ("Monomial", monomial_basis, None, "red"),
    ("Legendre", legendre_basis, None, "green"),
    ("Chebyshev", chebyshev_basis, chebyshev_weight, "blue")
]

# Maximum number of terms to show in animation
max_terms = 15

# Create frames for animation
for n in range(1, max_terms + 1):
    plt.figure(figsize=(10, 6))
    plt.title(f'L² Projections of |x| with {n} basis functions')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    
    # Plot true function
    plt.plot(x_eval, y_true, 'k-', linewidth=2, label='|x|')
    
    # Store projection data for verification
    projections_data = {}
    
    # Compute and plot each basis projection
    for name, basis_func, weight_func, color in basis_functions:
        try:
            # Compute L2 projection
            coeffs = compute_l2_projection(basis_func, n, weight_func)
            
            # Print coefficients for debugging
            print(f"\n{name} coefficients (n={n}):")
            for i, c in enumerate(coeffs):
                print(f"  c_{i} = {c:.8f}")
            
            # Evaluate projection
            y_proj = evaluate_l2_projection(x_eval, coeffs, basis_func)
            
            # Store for verification
            projections_data[name] = y_proj
            
            # Calculate L² error
            error = np.sqrt(np.mean((y_proj - y_true)**2))
            print(f"{name} L² Error: {error:.6f}")
            
            # Plot projection
            plt.plot(x_eval, y_proj, color=color, linewidth=1.5, label=f'{name} (n={n})')
            
            # Check for NaN or Inf values
            if np.any(np.isnan(y_proj)) or np.any(np.isinf(y_proj)):
                print(f"Warning: {name} projection contains NaN or Inf values!")
            
        except Exception as e:
            print(f"Error with {name} basis: {e}")
    
    plt.legend()
    plt.ylim(-0.1, 1.1)  # Fixed y-axis limits
    
    # Save the frame
    plt.savefig(f"l2_projection_frames/frame_{n:03d}.png", dpi=150)
    plt.close()
    
    # Create verification plot for this n value
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Individual L² Projections (n={n})')
    
    for i, (name, _, _, color) in enumerate(basis_functions):
        plt.subplot(1, 3, i+1)
        plt.title(name)
        plt.plot(x_eval, y_true, 'k-', linewidth=2, label='|x|')
        
        if name in projections_data:
            plt.plot(x_eval, projections_data[name], color=color, linewidth=1.5, label=f'{name}')
        else:
            plt.text(0, 0.5, "Projection failed", ha='center')
        
        plt.grid(True)
        plt.legend()
        plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"l2_projection_frames/verify_{n:03d}.png", dpi=150)
    plt.close()

print("Animation frames created in 'l2_projection_frames' directory.")
print("To create a GIF, you can use:")
print("ImageMagick: convert -delay 50 l2_projection_frames/frame_*.png l2_projections.gif")
print("or FFmpeg: ffmpeg -framerate 2 -i l2_projection_frames/frame_%03d.png l2_projections.gif")