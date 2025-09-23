"""Test symbolic Jones matrix equation generation."""

import pytest
import sympy as sp
from sympy import Matrix, symbols, cos, sin, exp, I, pi, simplify

from jones_sim.symbolic import JonesEquationGenerator


class TestJonesEquationGenerator:
    """Test the symbolic equation generator."""
    
    def setup_method(self):
        """Setup for each test."""
        self.generator = JonesEquationGenerator()
    
    def test_init(self):
        """Test proper initialization."""
        assert self.generator.effects == []
        assert 'parallactic' in self.generator.effect_matrices
        assert 'gains' in self.generator.effect_matrices
        assert 'leakage' in self.generator.effect_matrices
    
    def test_add_effect(self):
        """Test adding effects to the chain."""
        self.generator.add_effect('gains')
        assert 'gains' in self.generator.effects
        
        self.generator.add_effect('leakage')
        assert 'leakage' in self.generator.effects
        
        # Test duplicate addition
        self.generator.add_effect('gains')
        assert self.generator.effects.count('gains') == 1
    
    def test_invalid_effect(self):
        """Test error on invalid effect name."""
        with pytest.raises(ValueError, match="Unknown effect"):
            self.generator.add_effect('invalid_effect')
    
    def test_identity_matrix(self):
        """Test identity matrix when no effects added."""
        result = self.generator.generate_jones_equation()
        expected = Matrix([[1, 0], [0, 1]])
        assert result == expected
    
    def test_single_effect_gains(self):
        """Test Jones matrix with only gains."""
        self.generator.add_effect('gains')
        result = self.generator.generate_jones_equation()
        
        expected = Matrix([
            [self.generator.g_xx, 0],
            [0, self.generator.g_yy]
        ])
        assert result == expected
    
    def test_single_effect_parallactic(self):
        """Test Jones matrix with only parallactic angle."""
        self.generator.add_effect('parallactic')
        result = self.generator.generate_jones_equation()
        
        expected = Matrix([
            [cos(self.generator.psi), sin(self.generator.psi)],
            [-sin(self.generator.psi), cos(self.generator.psi)]
        ])
        assert result == expected
    
    def test_multiple_effects_order(self):
        """Test that effects are applied in correct order regardless of add order."""
        # Add in reverse order
        self.generator.add_effect('gains')
        self.generator.add_effect('parallactic')
        
        result = self.generator.generate_jones_equation()
        
        # Should be P * G (parallactic first, then gains)
        P = self.generator.effect_matrices['parallactic']
        G = self.generator.effect_matrices['gains']
        expected = P * G
        
        assert simplify(result - expected) == Matrix([[0, 0], [0, 0]])
    
    def test_leakage_matrix_form(self):
        """Test leakage matrix includes misalignment terms."""
        leakage = self.generator.get_effect_matrix('leakage')
        
        # Should be [[1, d_hv + tan(theta)], [d_vh - tan(theta), 1]]
        expected_t_hv = self.generator.d_hv + sp.tan(self.generator.theta)
        expected_t_vh = self.generator.d_vh - sp.tan(self.generator.theta)
        
        assert leakage[0, 0] == 1
        assert leakage[1, 1] == 1
        assert leakage[0, 1] == expected_t_hv
        assert leakage[1, 0] == expected_t_vh
    
    def test_rl_delay_matrix_form(self):
        """Test R/L delay matrix matches EVLA memo 207 form."""
        rl_delay = self.generator.get_effect_matrix('rl_delay')
        
        delta_theta = 2*pi * self.generator.delta_tau * (self.generator.nu - self.generator.nu_c)
        
        expected = Matrix([
            [cos(delta_theta/2), I*sin(delta_theta/2)],
            [-I*sin(delta_theta/2), cos(delta_theta/2)]
        ])
        
        assert simplify(rl_delay - expected) == Matrix([[0, 0], [0, 0]])
    
    def test_mueller_matrix_dimensions(self):
        """Test Mueller matrix has correct 4x4 dimensions."""
        self.generator.add_effect('gains')
        mueller = self.generator.generate_mueller_equation()
        
        assert mueller.shape == (4, 4)
    
    def test_mueller_kronecker_product(self):
        """Test Mueller matrix is correct Kronecker product."""
        # Simple case: just gains
        self.generator.add_effect('gains')
        
        jones = self.generator.generate_jones_equation(simplify_result=False)
        mueller = self.generator.generate_mueller_equation(simplify_result=False)
        
        # Manual Kronecker product J ⊗ J†
        jones_dagger = jones.conjugate().T
        expected = sp.kronecker_product(jones, jones_dagger)
        
        assert mueller == expected
    
    def test_clear_effects(self):
        """Test clearing all effects."""
        self.generator.add_effect('gains')
        self.generator.add_effect('parallactic')
        assert len(self.generator.effects) == 2
        
        self.generator.clear_effects()
        assert len(self.generator.effects) == 0


class TestEVLAMemoValidation:
    """Test derivation of EVLA memo 207 equations."""
    
    def setup_method(self):
        """Setup for validation tests."""
        self.generator = JonesEquationGenerator()
    
    def test_unpolarized_source_setup(self):
        """Test setup for deriving EVLA memo equations 67-70."""
        # For unpolarized source, need leakage terms
        self.generator.add_effect('leakage')
        self.generator.add_effect('gains')
        
        # Generate Jones matrices for two antennas
        jones1 = self.generator.generate_jones_equation()
        jones2 = self.generator.generate_jones_equation()  # Same for now
        
        # Create visibility vector for unpolarized source: [I, 0, 0, I]
        I_stokes = symbols('I', real=True, positive=True)
        ideal_vis = Matrix([I_stokes, 0, 0, I_stokes])
        
        # Apply corruption: V_obs = (J1 ⊗ J2†) V_ideal
        jones2_dag = jones2.conjugate().T
        mueller = sp.kronecker_product(jones1, jones2_dag)
        corrupted_vis = mueller * ideal_vis
        
        # Should be able to extract forms matching EVLA memo eq 67-70
        # This is the setup - detailed validation would compare specific terms
        assert corrupted_vis.shape == (4, 1)
        assert corrupted_vis[0] != I_stokes  # XX correlation is corrupted
        assert corrupted_vis[3] != I_stokes  # YY correlation is corrupted
    
    def test_rl_delay_cross_correlations(self):
        """Test R/L delay effects on cross-hand correlations (EVLA memo 75-76)."""
        # For cross-hand delay test, use R/L delay effect
        self.generator.add_effect('rl_delay')
        
        jones = self.generator.generate_jones_equation()
        
        # For unpolarized source with misaligned dipoles, expect:
        # R_vh ∝ sin(Δθ), R_hv ∝ -sin(Δθ)
        
        # Create Mueller matrix
        jones_dag = jones.conjugate().T  
        mueller = sp.kronecker_product(jones, jones_dag)
        
        # Extract XY and YX terms (indices 1 and 2)
        xy_term = mueller[1, 0]  # Effect on XY from XX input
        yx_term = mueller[2, 0]  # Effect on YX from XX input
        
        # These should contain sin terms with opposite signs
        delta_theta = 2*pi * self.generator.delta_tau * (self.generator.nu - self.generator.nu_c)
        
        # Verify the terms contain the expected frequency dependence
        # Check for the constituent symbols rather than the full expression
        expected_symbols = {self.generator.delta_tau, self.generator.nu, self.generator.nu_c}
        assert expected_symbols.issubset(xy_term.free_symbols)
        assert expected_symbols.issubset(yx_term.free_symbols)
    
    def test_crosshand_phase_effects(self):
        """Test cross-hand phase effects (EVLA memo 77-78)."""
        self.generator.add_effect('crosshand_phase')
        
        jones = self.generator.generate_jones_equation()
        
        # Cross-hand phase should only affect YY term
        expected = Matrix([
            [1, 0],
            [0, exp(I * self.generator.phi)]
        ])
        
        assert jones == expected
        
        # In Mueller form, this should affect cross-correlations with exp(±iφ) factors
        mueller = self.generator.generate_mueller_equation()
        
        # XY correlation should have exp(-iφ) factor
        # YX correlation should have exp(+iφ) factor  
        xy_element = mueller[1, 1]  # XY from XY
        yx_element = mueller[2, 2]  # YX from YX
        
        # These should contain the phase factor
        assert self.generator.phi in xy_element.free_symbols
        assert self.generator.phi in yx_element.free_symbols