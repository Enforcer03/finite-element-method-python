import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
import argparse
from matplotlib.widgets import Slider, RadioButtons, Button

def plot_quadratic_solution(x, u, args, ax):
    """Plot solution with proper quadratic interpolation"""
    # For quadratic basis, we need to plot with interpolation
    if args.basis == 'quadratic':
        # Create fine grid for smooth plotting
        x_fine = np.linspace(0, 1, 500)
        u_fine = np.zeros_like(x_fine)
        
        # Number of elements
        n = args.n
        
        # Loop through each element
        for i in range(n):
            # Find points in this element
            x_elem_start = i/n
            x_elem_end = (i+1)/n
            mask = (x_fine >= x_elem_start) & (x_fine <= x_elem_end)
            x_local = x_fine[mask]
            
            if len(x_local) == 0:
                continue
                
            # Map to reference element [-1, 1]
            xi = 2 * (x_local - x_elem_start) / (x_elem_end - x_elem_start) - 1
            
            # Get nodal values for this element
            nodes = [2*i, 2*i+1, 2*i+2]
            u_nodes = u[nodes]
            
            # Quadratic shape functions on reference element
            N0 = 0.5 * xi * (xi - 1)
            N1 = (1 - xi**2)
            N2 = 0.5 * xi * (xi + 1)
            
            # Interpolate
            u_fine[mask] = N0 * u_nodes[0] + N1 * u_nodes[1] + N2 * u_nodes[2]
        
        # Plot interpolated solution
        ax.plot(x_fine, u_fine, 'b-', linewidth=2)
    else:
        # For linear basis, just plot at nodes
        ax.plot(x, u, 'b-', linewidth=2)

# Replace the plotting call in main() function:
# Change from:
# ax1.plot(x, u, 'b-', linewidth=2)
# To:
# plot_quadratic_solution(x, u, args, ax1)

def solve_poisson_1d(n=100, f=lambda x: np.ones_like(x), bc_type='dirichlet', 
                     bc_values=(0,0), basis='linear'):
    """Optimized 1D Poisson solver"""
    # Mesh generation - vectorized
    if basis == 'linear':
        x = np.linspace(0, 1, n+1)
        dof = n+1
    else:  # quadratic
        x = np.linspace(0, 1, 2*n+1)
        dof = 2*n+1
    
    h = 1/n
    
    # Fast assembly using diagonals (for linear basis)
    if basis == 'linear':
        # Main diagonal: 2/h except at boundaries
        diag_main = np.ones(dof) * 2/h
        
        # Off-diagonal: -1/h
        diag_off = np.ones(dof-1) * -1/h
        
        # Create sparse matrix directly
        A = diags([diag_main, diag_off, diag_off], [0, -1, 1])
        
        # Load vector - vectorized integration
        # Use vectorized midpoint rule for efficiency
        x_mid = (x[:-1] + x[1:]) / 2
        f_mid = f(x_mid)
        
        # Distribute load to nodes
        b = np.zeros(dof)
        b[:-1] += h/2 * f_mid
        b[1:] += h/2 * f_mid
    
    else:  
        # Pre-allocate arrays for efficient assembly using LIL format instead of diags
        A = lil_matrix((dof, dof))
        b = np.zeros(dof)
        
        # Standard element matrices for quadratic basis
        A_e = np.array([
            [7/(3*h), -8/(3*h), 1/(3*h)],
            [-8/(3*h), 16/(3*h), -8/(3*h)],
            [1/(3*h), -8/(3*h), 7/(3*h)]
        ])
        
        # Loop over elements (still needed for quadratic basis)
        for i in range(n):
            node_indices = [2*i, 2*i+1, 2*i+2]
            
            # Element endpoints
            x1, x3 = x[2*i], x[2*i+2]
            
            # Three-point Gaussian quadrature points
            gp1, gp2, gp3 = x1 + 0.1127*h, x1 + 0.5*h, x1 + 0.8873*h
            f_gp = f(np.array([gp1, gp2, gp3]))
            
            # Simplified load vector computation
            b_e = h/6 * np.array([
                f_gp[0] + 0.5*f_gp[1] + 0.1*f_gp[2],
                0.8*f_gp[0] + 2*f_gp[1] + 0.8*f_gp[2],
                0.1*f_gp[0] + 0.5*f_gp[1] + f_gp[2]
            ])
            
            # Fast assembly using indexing
            for j in range(3):
                for k in range(3):
                    A[node_indices[j], node_indices[k]] += A_e[j, k]
                b[node_indices[j]] += b_e[j]
    
    # Apply boundary conditions - all vectorized
    if bc_type == 'dirichlet':
        # u(0) = bc_values[0], u(1) = bc_values[1]
        A = A.tolil()  # Convert to LIL for efficient row operations
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = bc_values[0]
        
        A[-1, :] = 0
        A[-1, -1] = 1
        b[-1] = bc_values[1]
        A = A.tocsr()  # Convert back to CSR for efficient solving
    
    elif bc_type == 'neumann':
        # u'(0) = bc_values[0], u'(1) = bc_values[1]
        b[0] -= bc_values[0]
        b[-1] += bc_values[1]
        
        # Pin one point to avoid singular matrix
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0
        A = A.tocsr()
    
    elif bc_type == 'mixed':
        # u(0) = bc_values[0], u'(1) = bc_values[1]
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = bc_values[0]
        b[-1] += bc_values[1]
        A = A.tocsr()
    
    # Solve system efficiently
    u = spsolve(A, b)
    
    return x, u

# Source functions dictionary
SOURCE_FUNCTIONS = {
    'sin': lambda x: 4*np.pi**2*np.sin(2*np.pi*x),
    'constant': lambda x: np.ones_like(x),
    'linear': lambda x: x,
    'quadratic': lambda x: x**2
}

def get_exact_solution(x, source_term, bc_type, bc_left, bc_right):
    """Return exact solution if available"""
    if bc_type == 'dirichlet' and bc_left == 0 and bc_right == 0:
        if source_term == 'sin':
            # The exact solution to -u''(x) = 4π²sin(2πx) with u(0)=u(1)=0 
            # is u(x) = sin(2πx)
            return np.sin(2*np.pi*x)
        elif source_term == 'constant':
            # For -u''(x) = 1 with u(0)=u(1)=0, the solution is u(x) = 0.5*x*(1-x)
            return 0.5*x*(1-x)
        elif source_term == 'linear':
            # For -u''(x) = x with u(0)=u(1)=0, the solution is u(x) = (1/6)*x*(1-x)*(1+2*x)
            return (1/6)*x*(1-x)*(1+2*x)
        elif source_term == 'quadratic':
            # For -u''(x) = x² with u(0)=u(1)=0, the solution is u(x) = (1/12)*x*(1-x)*(x²-x+1)
            return (1/12)*x*(1-x)*(x**2-x+1)
    
    # For other cases, we can try to compute a numerical approximation
    elif source_term == 'sin' and bc_type == 'neumann' and bc_left == 0 and bc_right == 0:
        # Exact solution for -u''(x) = 4π²sin(2πx) with u'(0)=u'(1)=0
        # This is a bit more complex as we have an undetermined constant
        return np.sin(2*np.pi*x) - np.sin(2*np.pi) * x
    
    # Return None if no exact solution is available
    return None

def custom_exact_solution(source_term, bc_type, bc_left, bc_right):
    """Generate a custom exact solution function if possible"""
    # Equation: -u''(x) = f(x)
    
    if source_term == 'sin':
        # f(x) = 4π²sin(2πx)
        if bc_type == 'dirichlet' and bc_left == 0 and bc_right == 0:
            # Most direct case
            return lambda x: np.sin(2*np.pi*x)
    
    elif source_term == 'constant':
        # f(x) = 1
        if bc_type == 'dirichlet' and bc_left == 0 and bc_right == 0:
            return lambda x: 0.5*x*(1-x)
        elif bc_type == 'dirichlet':
            # General case: u(x) = 0.5*x*(1-x) + bc_left*(1-x) + bc_right*x
            return lambda x: 0.5*x*(1-x) + bc_left*(1-x) + bc_right*x
    
    # For other cases, return None
    return None

def interactive_plot():
    """Create an interactive matplotlib plot"""
    # Initial values
    n_init = 50
    bc_type_init = 'dirichlet'
    bc_left_init = 0
    bc_right_init = 0
    source_term_init = 'sin'
    basis_init = 'linear'
    
    # Solve with initial values
    x, u = solve_poisson_1d(
        n=n_init,
        f=SOURCE_FUNCTIONS[source_term_init],
        bc_type=bc_type_init,
        bc_values=(bc_left_init, bc_right_init),
        basis=basis_init
    )
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)
    
    # Initial plot
    line_fem, = ax.plot(x, u, 'b-', linewidth=2, label='FEM Solution')
    
    # Get exact solution if available
    exact_func = custom_exact_solution(source_term_init, bc_type_init, bc_left_init, bc_right_init)
    
    # Create text annotation for formula (initially empty)
    formula_text = ax.text(0.98, 0.98, "", transform=ax.transAxes, 
                          ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Display exact solution and formula if available
    if exact_func is not None:
        exact = exact_func(x)
        line_exact, = ax.plot(x, exact, 'r--', linewidth=2, label='Exact Solution')
        
        # Set formula text based on source term
        if source_term_init == 'sin':
            formula_text.set_text('Exact: $u(x) = \\sin(2\\pi x)$')
        elif source_term_init == 'constant':
            if bc_left_init == 0 and bc_right_init == 0:
                formula_text.set_text('Exact: $u(x) = \\frac{1}{2}x(1-x)$')
            else:
                formula_text.set_text(f'Exact: $u(x) = \\frac{{1}}{{2}}x(1-x) + {bc_left_init}(1-x) + {bc_right_init}x$')
        elif source_term_init == 'linear':
            formula_text.set_text('Exact: $u(x) = \\frac{1}{6}x(1-x)(1+2x)$')
        elif source_term_init == 'quadratic':
            formula_text.set_text('Exact: $u(x) = \\frac{1}{12}x(1-x)(x^2-x+1)$')
        
        ax.legend()
    
    ax.grid(True)
    ax.set_title(f'1D Poisson Equation: {bc_type_init} BCs, {basis_init} basis')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')

    # Add sliders and radio buttons
    ax_n = plt.axes([0.1, 0.25, 0.65, 0.03])
    ax_left = plt.axes([0.1, 0.2, 0.65, 0.03])
    ax_right = plt.axes([0.1, 0.15, 0.65, 0.03])
    ax_bc_type = plt.axes([0.1, 0.05, 0.2, 0.1])
    ax_source = plt.axes([0.4, 0.05, 0.2, 0.1])
    ax_basis = plt.axes([0.7, 0.05, 0.2, 0.1])
    
    slider_n = Slider(ax_n, 'Elements', 10, 200, valinit=n_init, valstep=10)
    slider_left = Slider(ax_left, 'Left BC', -2, 2, valinit=bc_left_init)
    slider_right = Slider(ax_right, 'Right BC', -2, 2, valinit=bc_right_init)
    
    radio_bc = RadioButtons(ax_bc_type, ('dirichlet', 'neumann', 'mixed'), active=0)
    radio_source = RadioButtons(ax_source, ('sin', 'constant', 'linear', 'quadratic'), active=0)
    radio_basis = RadioButtons(ax_basis, ('linear', 'quadratic'), active=0)
    
    # Update function
    def update(val=None):
        # Get current values
        n = int(slider_n.val)
        bc_type = radio_bc.value_selected
        bc_left = slider_left.val
        bc_right = slider_right.val
        source_term = radio_source.value_selected
        basis = radio_basis.value_selected
        
        # Solve with new values
        x, u = solve_poisson_1d(
            n=n,
            f=SOURCE_FUNCTIONS[source_term],
            bc_type=bc_type,
            bc_values=(bc_left, bc_right),
            basis=basis
        )
        
        # Update FEM solution
        line_fem.set_xdata(x)
        line_fem.set_ydata(u)
        
        # Update exact solution if available
        exact_func = custom_exact_solution(source_term, bc_type, bc_left, bc_right)
        
        # Check if exact solution line exists
        exact_line_exists = len(ax.lines) > 1
        
        if exact_func is not None:
            exact = exact_func(x)
            if exact_line_exists:
                ax.lines[1].set_xdata(x)
                ax.lines[1].set_ydata(exact)
                ax.lines[1].set_visible(True)
            else:
                line_exact, = ax.plot(x, exact, 'r--', linewidth=2, label='Exact Solution')
                ax.legend()
            
            # Update formula text based on source term and BCs
            if source_term == 'sin':
                formula_text.set_text('Exact: $u(x) = \\sin(2\\pi x)$')
            elif source_term == 'constant':
                if bc_left == 0 and bc_right == 0:
                    formula_text.set_text('Exact: $u(x) = \\frac{1}{2}x(1-x)$')
                else:
                    formula_text.set_text(f'Exact: $u(x) = \\frac{{1}}{{2}}x(1-x) + {bc_left}(1-x) + {bc_right}x$')
            elif source_term == 'linear':
                formula_text.set_text('Exact: $u(x) = \\frac{1}{6}x(1-x)(1+2x)$')
            elif source_term == 'quadratic':
                formula_text.set_text('Exact: $u(x) = \\frac{1}{12}x(1-x)(x^2-x+1)$')
            
            formula_text.set_visible(True)
        else:
            if exact_line_exists:
                ax.lines[1].set_visible(False)
            formula_text.set_visible(False)
        
        # Update title
        ax.set_title(f'1D Poisson Equation: {bc_type} BCs, {basis} basis')
        
        # Rescale and draw
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
    
    # Connect update function to widgets
    slider_n.on_changed(update)
    slider_left.on_changed(update)
    slider_right.on_changed(update)
    radio_bc.on_clicked(update)
    radio_source.on_clicked(update)
    radio_basis.on_clicked(update)
    
    plt.show()

def main():
    """Main function to run from command line"""
    parser = argparse.ArgumentParser(description='1D Poisson Equation Solver')
    parser.add_argument('--n', type=int, default=50, help='Number of elements')
    parser.add_argument('--bc', type=str, default='dirichlet', 
                        choices=['dirichlet', 'neumann', 'mixed'],
                        help='Boundary condition type')
    parser.add_argument('--left', type=float, default=0, help='Left boundary value')
    parser.add_argument('--right', type=float, default=0, help='Right boundary value')
    parser.add_argument('--source', type=str, default='sin',
                        choices=['sin', 'constant', 'linear', 'quadratic'],
                        help='Source function')
    parser.add_argument('--basis', type=str, default='linear',
                        choices=['linear', 'quadratic'],
                        help='Basis function type')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_plot()
    else:
        # Solve with provided arguments
        x, u = solve_poisson_1d(
            n=args.n,
            f=SOURCE_FUNCTIONS[args.source],
            bc_type=args.bc,
            bc_values=(args.left, args.right),
            basis=args.basis
        )
        
        # Plot solution - create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # FEM solution on the left
        if args.basis == 'quadratic':
            plot_quadratic_solution(x, u, args, ax1)
        else:
            ax1.plot(x, u, 'b-', linewidth=2)
        ax1.set_title('FEM Solution')
        ax1.grid(True)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        
        # Make sure y-axis limits are the same for both plots
        y_min = min(u)
        y_max = max(u)
        margin = (y_max - y_min) * 0.1  # Add 10% margin
        ax1.set_ylim(y_min - margin, y_max + margin)
        
        # Create a higher resolution x-grid for exact solution
        x_exact = np.linspace(0, 1, 500)  # 500 points for smooth curve
        
        # Compute exact solutions based on source term and boundary conditions
        exact = None
        formula_text = ""
        
        if args.source == 'sin' and args.bc == 'dirichlet' and args.left == 0 and args.right == 0:
            # For -u''(x) = 4π²sin(2πx) with u(0)=u(1)=0
            exact = np.sin(2*np.pi*x_exact)
            formula_text = r'$u(x) = \sin(2\pi x)$'
        elif args.source == 'constant' and args.bc == 'dirichlet' and args.left == 0 and args.right == 0:
            # For -u''(x) = 1 with u(0)=u(1)=0
            exact = 0.5*x_exact*(1-x_exact)
            formula_text = r'$u(x) = \frac{1}{2}x(1-x)$'
        elif args.source == 'constant' and args.bc == 'dirichlet':
            # For -u''(x) = 1 with general Dirichlet BCs
            exact = 0.5*x_exact*(1-x_exact) + args.left*(1-x_exact) + args.right*x_exact
            formula_text = f'$u(x) = \\frac{{1}}{{2}}x(1-x) + {args.left}(1-x) + {args.right}x$'
        elif args.source == 'linear' and args.bc == 'dirichlet' and args.left == 0 and args.right == 0:
            # For -u''(x) = x with u(0)=u(1)=0
            exact = (1/6)*x_exact*(1-x_exact)*(1+2*x_exact)
            formula_text = r'$u(x) = \frac{1}{6}x(1-x)(1+2x)$'
        elif args.source == 'quadratic' and args.bc == 'dirichlet' and args.left == 0 and args.right == 0:
            # For -u''(x) = x² with u(0)=u(1)=0
            exact = (1/12)*x_exact*(1-x_exact)*(x_exact**2-x_exact+1)
            formula_text = r'$u(x) = \frac{1}{12}x(1-x)(x^2-x+1)$'
            
        # PDE equation text based on source term
        pde_text = ""
        if args.source == 'sin':
            pde_text = r'$-u^{\prime\prime}(x) = 4\pi^2\sin(2\pi x)$'
        elif args.source == 'constant':
            pde_text = r'$-u^{\prime\prime}(x) = 1$'
        elif args.source == 'linear':
            pde_text = r'$-u^{\prime\prime}(x) = x$'
        elif args.source == 'quadratic':
            pde_text = r'$-u^{\prime\prime}(x) = x^2$'
        
        # Boundary conditions text
        bc_text = ""
        if args.bc == 'dirichlet':
            bc_text = f'$u(0) = {args.left}$, $u(1) = {args.right}$'
        elif args.bc == 'neumann':
            bc_text = f'$u^\\prime(0) = {args.left}$, $u^\\prime(1) = {args.right}$'
        elif args.bc == 'mixed':
            bc_text = f'$u(0) = {args.left}$, $u^\\prime(1) = {args.right}$'
        
        # Add parameter information for the info box
        params_text = f"""PDE: {pde_text}
Boundary conditions: {bc_text}
Mesh size: {args.n} elements
Basis functions: {args.basis}"""
        
        # Plot exact solution on the right if available
        if exact is not None:
            ax2.plot(x_exact, exact, 'r-', linewidth=2)
            ax2.set_title('Exact Solution')
            
            # Display formula in upper right corner of exact solution plot
            ax2.text(0.98, 0.98, formula_text, transform=ax2.transAxes,
                    ha='right', va='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8))
            
            # Make sure y-axis limits match the FEM solution
            ax2.set_ylim(y_min - margin, y_max + margin)
        else:
            ax2.text(0.5, 0.5, "No exact solution available", 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Exact Solution')
            ax2.axis('off')
        
        ax2.grid(True)
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        
        # Add parameter info box on the right side
        plt.figtext(0.98, 0.5, params_text, ha='right', va='center', 
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'1D Poisson Equation: {args.bc} BCs, {args.basis} basis')
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Make space for the parameter text
        plt.show()

if __name__ == '__main__':
    main()