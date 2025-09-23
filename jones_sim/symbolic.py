"""Symbolic Jones matrix equation generation using SymPy."""

from typing import List, Dict, Optional
import sympy as sp
from sympy import Matrix, symbols, cos, sin, exp, I, pi, simplify


class JonesEquationGenerator:
    """Generate symbolic Jones matrix equations using SymPy.
    
    Takes user-specified effects and outputs symbolic equations.
    Handles matrix multiplication and simplification.
    Can generate Mueller matrices via Kronecker product.
    """
    
    def __init__(self):
        self.effects: List[str] = []
        self.effect_matrices: Dict[str, Matrix] = {}
        self._define_symbols()
        self._define_effect_matrices()
    
    def _define_symbols(self):
        """Define all symbolic parameters used in Jones matrices."""
        # Parallactic angle
        self.psi = symbols('psi', real=True)
        
        # Electronic gains (complex)
        self.g_xx, self.g_yy = symbols('g_xx g_yy', complex=True)
        
        # Leakage terms
        self.d_hv, self.d_vh = symbols('d_hv d_vh', complex=True)
        self.theta = symbols('theta', real=True)  # misalignment angle
        
        # Bandpass/delay parameters
        self.tau_xx, self.tau_yy = symbols('tau_xx tau_yy', real=True)
        self.nu, self.nu_c = symbols('nu nu_c', real=True, positive=True)
        
        # TEC parameters  
        self.t_xx, self.t_yy = symbols('t_xx t_yy', real=True)
        self.theta_xx, self.theta_yy = symbols('theta_xx theta_yy', real=True)
        self.nu_min, self.nu_max = symbols('nu_min nu_max', real=True, positive=True)
        
        # R/L delay difference
        self.delta_tau = symbols('delta_tau', real=True)
        
        # Cross-hand phase offset
        self.phi = symbols('phi', real=True)
    
    def _define_effect_matrices(self):
        """Define symbolic matrices for each Jones effect."""
        
        # Parallactic angle rotation
        # Note: Using convention from jones_measurement_equation_math.tex
        # Alternative convention in jones_measurement_equation.tex: [[cos(ψ), -sin(ψ)], [sin(ψ), cos(ψ)]]
        self.effect_matrices['parallactic'] = Matrix([
            [cos(self.psi), sin(self.psi)],
            [-sin(self.psi), cos(self.psi)]
        ])
        
        # Electronic gains (diagonal)
        self.effect_matrices['gains'] = Matrix([
            [self.g_xx, 0],
            [0, self.g_yy]
        ])
        
        # Combined leakage and misalignment  
        T_hv = self.d_hv + sp.tan(self.theta)
        T_vh = self.d_vh - sp.tan(self.theta)
        self.effect_matrices['leakage'] = Matrix([
            [1, T_hv],
            [T_vh, 1]
        ])
        
        # Bandpass/delay effects
        self.effect_matrices['bandpass'] = Matrix([
            [exp(I * 2*pi * self.tau_xx * (self.nu - self.nu_c)), 0],
            [0, exp(I * 2*pi * self.tau_yy * (self.nu - self.nu_c))]
        ])
        
        # TEC effects
        alpha_xx = 2*pi * self.t_xx * (1/self.nu + 
                   (sp.log(self.nu_min) - sp.log(self.nu_max))/(self.nu_max - self.nu_min)) + self.theta_xx
        alpha_yy = 2*pi * self.t_yy * (1/self.nu + 
                   (sp.log(self.nu_min) - sp.log(self.nu_max))/(self.nu_max - self.nu_min)) + self.theta_yy
        self.effect_matrices['tec'] = Matrix([
            [exp(I * alpha_xx), 0],
            [0, exp(I * alpha_yy)]
        ])
        
        # R/L delay difference
        delta_theta = 2*pi * self.delta_tau * (self.nu - self.nu_c)
        self.effect_matrices['rl_delay'] = Matrix([
            [cos(delta_theta/2), I*sin(delta_theta/2)],
            [-I*sin(delta_theta/2), cos(delta_theta/2)]
        ])
        
        # Cross-hand phase offset
        self.effect_matrices['crosshand_phase'] = Matrix([
            [1, 0],
            [0, exp(I * self.phi)]
        ])
        
        # Identity matrix
        self.effect_matrices['identity'] = Matrix([
            [1, 0],
            [0, 1]
        ])
    
    def add_effect(self, effect_name: str):
        """Add an effect to the Jones chain.
        
        Args:
            effect_name: One of 'parallactic', 'gains', 'leakage', 'bandpass', 
                        'tec', 'rl_delay', 'crosshand_phase'
        """
        if effect_name not in self.effect_matrices:
            raise ValueError(f"Unknown effect: {effect_name}. "
                           f"Available: {list(self.effect_matrices.keys())}")
        
        if effect_name not in self.effects:
            self.effects.append(effect_name)
    
    def generate_jones_equation(self, simplify_result: bool = True) -> Matrix:
        """Generate the complete Jones matrix equation.
        
        Multiplies all added effects in the standard order:
        J = P * E * X * G * B * T * R * C
        
        Missing effects are replaced with identity matrices.
        
        Args:
            simplify_result: Whether to apply SymPy simplification
            
        Returns:
            2x2 symbolic Jones matrix
        """
        # Standard order from EVLA memo 207
        standard_order = [
            'parallactic',     # P
            'elevation',       # E (not implemented yet)
            'leakage',         # X  
            'gains',           # G
            'bandpass',        # B
            'tec',             # T
            'rl_delay',        # R
            'crosshand_phase'  # C
        ]
        
        result = self.effect_matrices['identity']
        
        for effect in standard_order:
            if effect in self.effects:
                if effect == 'elevation':
                    # Placeholder for elevation effects
                    continue
                result = result * self.effect_matrices[effect]
        
        if simplify_result:
            result = simplify(result)
            
        return result
    
    def generate_mueller_equation(self, antenna1_jones: Optional[Matrix] = None,
                                antenna2_jones: Optional[Matrix] = None,
                                simplify_result: bool = True) -> Matrix:
        """Generate Mueller matrix via Kronecker product.
        
        Implements: M = J1 ⊗ J2†
        
        Args:
            antenna1_jones: Jones matrix for antenna 1 (default: use generated)
            antenna2_jones: Jones matrix for antenna 2 (default: use generated) 
            simplify_result: Whether to apply SymPy simplification
            
        Returns:
            4x4 symbolic Mueller matrix operating on [XX, XY, YX, YY]
        """
        if antenna1_jones is None:
            antenna1_jones = self.generate_jones_equation(simplify_result=False)
        if antenna2_jones is None:
            antenna2_jones = self.generate_jones_equation(simplify_result=False)
            
        # Kronecker product: J1 ⊗ J2†
        j2_dagger = antenna2_jones.conjugate().T
        mueller = sp.kronecker_product(antenna1_jones, j2_dagger)
        
        if simplify_result:
            mueller = simplify(mueller)
            
        return mueller
    
    def get_effect_matrix(self, effect_name: str) -> Matrix:
        """Get the symbolic matrix for a specific effect.
        
        Args:
            effect_name: Name of the effect
            
        Returns:
            2x2 symbolic matrix for the effect
        """
        if effect_name not in self.effect_matrices:
            raise ValueError(f"Unknown effect: {effect_name}")
        return self.effect_matrices[effect_name]
    
    def clear_effects(self):
        """Clear all added effects."""
        self.effects.clear()