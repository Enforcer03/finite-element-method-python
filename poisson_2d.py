#!/usr/bin/env python3
"""
Complete 2D Poisson Equation Solver Using scikit-FEM
Usage:
    python poisson_2d_complete.py --n=4 --bc=dirichlet --source=sin --interactive

Options:
    --n: Refinement level for the mesh (default: 4)
    --bc: Boundary condition type (dirichlet, neumann, mixed) (default: dirichlet)
    --source: Source function (sin, constant, linear, quadratic, custom) (default: sin)
    --interactive: Run in interactive mode
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import argparse

# Import scikit-FEM modules
from skfem import *
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot, draw

def create_mesh(refine_level):
    """Create mesh with specified refinement level."""
    mesh = MeshTri()
    for _ in range(refine_level):
        mesh = mesh.refined()
    return mesh

def solve_poisson_2d(refine_level=4, bc_type='dirichlet', source_term='sin'):
    """Solve 2D Poisson equation using scikit-FEM."""
    # Create mesh with specified refinement
    mesh = create_mesh(refine_level)
    
    # Define element and basis
    element = ElementTriP1()
    basis = Basis(mesh, element)
    
    # Define the Laplacian operator
    @BilinearForm
    def laplace(u, v, w):
        return dot(grad(u), grad(v))
    
    # Define source term function
    if source_term == 'sin':
        @LinearForm
        def source(v, w):
            x, y = w.x
            return v * (8 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y))
    
    elif source_term == 'constant':
        @LinearForm
        def source(v, w):
            return v
    
    elif source_term == 'linear':
        @LinearForm
        def source(v, w):
            x, y = w.x
            return v * (x + y)
    
    elif source_term == 'quadratic':
        @LinearForm
        def source(v, w):
            x, y = w.x
            return v * (x**2 + y**2)
    
    elif source_term == 'custom':
        @LinearForm
        def source(v, w):
            x, y = w.x
            # You can define any custom source function here
            return v * np.exp(-10*((x-0.5)**2 + (y-0.5)**2))
    
    # Assemble system
    A = laplace.assemble(basis)
    b = source.assemble(basis)
    
    # Apply boundary conditions
    if bc_type == 'dirichlet':
        # u = 0 on all boundaries
        dofs = basis.get_dofs()
        A, b = enforce(A, b, D=dofs)
    
    elif bc_type == 'neumann':
        # Natural boundary condition (du/dn = 0)
        # Fix one point to avoid singular matrix
        dofs = basis.get_dofs(lambda x: np.abs(x[0])**2 + np.abs(x[1])**2 < 1e-10)
        A, b = enforce(A, b, D=dofs)
    
    elif bc_type == 'mixed':
        # Dirichlet on bottom and left, Neumann on top and right
        dofs = basis.get_dofs(lambda x: (np.abs(x[0]) < 1e-10) | (np.abs(x[1]) < 1e-10))
        A, b = enforce(A, b, D=dofs)
    
    # Solve the system
    x = solve(A, b)
    
    return mesh, basis, x

def get_exact_solution(mesh, source_term, bc_type):
    """Return exact solution function if available."""
    if bc_type == 'dirichlet' and source_term == 'sin':
        # For -∇²u = 8π²sin(2πx)sin(2πy) with u=0 on boundary
        def exact_func(p):
            return np.sin(2*np.pi*p[0]) * np.sin(2*np.pi*p[1])
        return exact_func
    
    elif bc_type == 'dirichlet' and source_term == 'constant':
        # Approximation for -∇²u = 1 with u=0 on boundary
        def exact_func(p):
            x, y = p
            return (x * (1-x)) * (y * (1-y))
        return exact_func
    
    return None

def plot_solution(mesh, basis, x, ax, title="FEM Solution"):
    """Plot solution on the given axis."""
    # Clear the axis
    ax.clear()
    
    # Plot the solution
    plot(basis, x, ax=ax, shading='gouraud', colorbar=True)
    
    # Set title and make square aspect ratio
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return ax

def plot_exact(mesh, exact_func, ax, title="Exact Solution", levels=None):
    """Plot exact solution on the given axis."""
    # If no exact solution is available
    if exact_func is None:
        ax.clear()
        ax.text(0.5, 0.5, "No exact solution available", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_aspect('equal')
        return None
    
    # Clear the axis
    ax.clear()
    
    # Create a grid for visualization
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    points = np.vstack((xx.flatten(), yy.flatten()))
    values = exact_func(points)
    values = values.reshape(xx.shape)
    
    # Plot the exact solution
    contour = ax.contourf(xx, yy, values, levels=levels, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax)
    
    # Set title and make square aspect ratio
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return cbar

def interactive_plot():
    """Create an interactive matplotlib plot."""
    # Initial values
    refine_level_init = 4
    bc_type_init = 'dirichlet'
    source_term_init = 'sin'
    
    # Solve with initial values
    mesh, basis, u = solve_poisson_2d(
        refine_level=refine_level_init,
        bc_type=bc_type_init,
        source_term=source_term_init
    )
    
    # Create figure for interactive plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)
    
    # Initial plots
    cbar1 = plot_solution(mesh, basis, u, ax1, title=f'FEM Solution: {bc_type_init} BCs')
    
    # Get exact solution if available
    exact_func = get_exact_solution(mesh, source_term_init, bc_type_init)
    cbar2 = plot_exact(mesh, exact_func, ax2)
    
    # Add parameter information
    eqn_text = ""
    if source_term_init == 'sin':
        eqn_text = r'$-\nabla^2 u = 8\pi^2\sin(2\pi x)\sin(2\pi y)$'
    elif source_term_init == 'constant':
        eqn_text = r'$-\nabla^2 u = 1$'
    elif source_term_init == 'linear':
        eqn_text = r'$-\nabla^2 u = x + y$'
    elif source_term_init == 'quadratic':
        eqn_text = r'$-\nabla^2 u = x^2 + y^2$'
    elif source_term_init == 'custom':
        eqn_text = r'$-\nabla^2 u = e^{-10((x-0.5)^2 + (y-0.5)^2)}$'
    
    # Create parameter info box
    params_text = f"""PDE: {eqn_text}
BC type: {bc_type_init}
Refinement level: {refine_level_init}"""
    
    info_text = fig.text(0.5, 0.96, params_text, ha='center', va='top', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add sliders and radio buttons
    ax_refine = plt.axes([0.1, 0.2, 0.8, 0.03])
    ax_bc_type = plt.axes([0.1, 0.1, 0.25, 0.1])
    ax_source = plt.axes([0.5, 0.1, 0.25, 0.1])
    
    slider_refine = Slider(ax_refine, 'Refinement', 1, 6, valinit=refine_level_init, valstep=1)
    radio_bc = RadioButtons(ax_bc_type, ('dirichlet', 'neumann', 'mixed'), active=0)
    radio_source = RadioButtons(ax_source, ('sin', 'constant', 'linear', 'quadratic', 'custom'), active=0)
    
    # Update function
    def update(val=None):
        # Get current values
        refine_level = int(slider_refine.val)
        bc_type = radio_bc.value_selected
        source_term = radio_source.value_selected
        
        # Solve with new values
        mesh, basis, u = solve_poisson_2d(
            refine_level=refine_level,
            bc_type=bc_type,
            source_term=source_term
        )
        
        # Update plots
        nonlocal cbar1, cbar2
        
        # Remove old colorbars if they exist
        if cbar1 is not None and hasattr(cbar1, 'remove'):
            cbar1.remove()
        if cbar2 is not None and hasattr(cbar2, 'remove'):
            cbar2.remove()
        
        # Plot new FEM solution
        cbar1 = plot_solution(mesh, basis, u, ax1, title=f'FEM Solution: {bc_type} BCs')
        
        # Get new exact solution if available
        exact_func = get_exact_solution(mesh, source_term, bc_type)
        
        # Plot exact solution with matching levels if possible
        if cbar1 is not None:
            levels = np.linspace(u.min(), u.max(), 30)
            cbar2 = plot_exact(mesh, exact_func, ax2, levels=levels)
        else:
            cbar2 = plot_exact(mesh, exact_func, ax2)
        
        # Update equation text
        eqn_text = ""
        if source_term == 'sin':
            eqn_text = r'$-\nabla^2 u = 8\pi^2\sin(2\pi x)\sin(2\pi y)$'
        elif source_term == 'constant':
            eqn_text = r'$-\nabla^2 u = 1$'
        elif source_term == 'linear':
            eqn_text = r'$-\nabla^2 u = x + y$'
        elif source_term == 'quadratic':
            eqn_text = r'$-\nabla^2 u = x^2 + y^2$'
        elif source_term == 'custom':
            eqn_text = r'$-\nabla^2 u = e^{-10((x-0.5)^2 + (y-0.5)^2)}$'
        
        # Update parameter info box
        params_text = f"""PDE: {eqn_text}
BC type: {bc_type}
Refinement level: {refine_level}"""
        
        info_text.set_text(params_text)
        
        fig.canvas.draw_idle()
    
    # Connect update function to widgets
    slider_refine.on_changed(update)
    radio_bc.on_clicked(update)
    radio_source.on_clicked(update)
    
    plt.show()

def main():
    """Main function to run from command line."""
    parser = argparse.ArgumentParser(description='2D Poisson Equation Solver using scikit-FEM')
    parser.add_argument('--n', type=int, default=4, help='Mesh refinement level')
    parser.add_argument('--bc', type=str, default='dirichlet', 
                        choices=['dirichlet', 'neumann', 'mixed'],
                        help='Boundary condition type')
    parser.add_argument('--source', type=str, default='sin',
                        choices=['sin', 'constant', 'linear', 'quadratic', 'custom'],
                        help='Source function')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_plot()
    else:
        # Solve with provided arguments
        mesh, basis, u = solve_poisson_2d(
            refine_level=args.n,
            bc_type=args.bc,
            source_term=args.source
        )
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot FEM solution
        print("Plotting FEM solution...")
        plot_solution(mesh, basis, u, ax1, title=f'FEM Solution: {args.bc} BCs')
        
        # Get exact solution if available
        exact_func = get_exact_solution(mesh, args.source, args.bc)
        
        # Plot exact solution with matching levels if possible
        levels = np.linspace(u.min(), u.max(), 30)
        plot_exact(mesh, exact_func, ax2, levels=levels)
        
        # Add parameter information
        eqn_text = ""
        if args.source == 'sin':
            eqn_text = r'$-\nabla^2 u = 8\pi^2\sin(2\pi x)\sin(2\pi y)$'
        elif args.source == 'constant':
            eqn_text = r'$-\nabla^2 u = 1$'
        elif args.source == 'linear':
            eqn_text = r'$-\nabla^2 u = x + y$'
        elif args.source == 'quadratic':
            eqn_text = r'$-\nabla^2 u = x^2 + y^2$'
        elif args.source == 'custom':
            eqn_text = r'$-\nabla^2 u = e^{-10((x-0.5)^2 + (y-0.5)^2)}$'
        
        # Create parameter info box
        params_text = f"""PDE: {eqn_text}
BC type: {args.bc}
Refinement level: {args.n}"""
        
        plt.figtext(0.98, 0.5, params_text, ha='right', va='center', 
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make space for the parameter text
        plt.show()

if __name__ == '__main__':
    main()