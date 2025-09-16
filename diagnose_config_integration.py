#!/usr/bin/env python3
"""
Diagnostic script to investigate configuration integration issues.

This script tests:
1. How configuration is loaded and passed to feature modules
2. Whether configuration changes actually affect feature output
3. Which specific configuration parameters are working vs not working
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.feature_union import build_feature_matrix
from volume_price_trade.features.ta_basic import compute_ta_features
from volume_price_trade.features.volume_profile import compute_volume_profile_features
from volume_price_trade.features.vpa import compute_vpa_features
from volume_price_trade.features.ict import compute_ict_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigIntegrationTester:
    """Tester for configuration integration issues."""
    
    def __init__(self):
        self.base_config = self._load_base_config()
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration."""
        try:
            config_path = Path("config/base.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def create_sample_data(self, n_bars: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        logger.info(f"Creating sample OHLCV data with {n_bars} bars")
        
        # Create datetime index
        start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')
        dates = pd.date_range(start=start_date, periods=n_bars, freq='5min')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 100.0
        
        # Generate price movements
        returns = np.random.normal(0, 0.001, n_bars)
        prices = [base_price]
        
        for i in range(1, n_bars):
            price_change = returns[i]
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Create OHLCV data
        data = {
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n_bars).astype(int)
        }
        
        # Ensure high >= low and high/low include open/close
        for i in range(n_bars):
            data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
            data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_config_loading(self) -> Dict[str, Any]:
        """Test how configuration is loaded and passed to modules."""
        logger.info("Testing configuration loading mechanism")
        
        result = {
            'success': True,
            'base_config_loaded': False,
            'config_structure': {},
            'issues': []
        }
        
        try:
            # Check if base config is loaded correctly
            if self.base_config and isinstance(self.base_config, dict):
                result['base_config_loaded'] = True
                logger.info("Base configuration loaded successfully")
                
                # Check key configuration sections
                key_sections = ['features', 'data', 'sessions', 'risk']
                for section in key_sections:
                    if section in self.base_config:
                        result['config_structure'][section] = list(self.base_config[section].keys())
                        logger.info(f"Config section '{section}' found with keys: {result['config_structure'][section]}")
                    else:
                        result['issues'].append(f"Missing config section: {section}")
                
                # Check specific feature parameters
                feature_params = self.base_config.get('features', {})
                key_params = ['atr_window', 'rvol_windows', 'volume_profile', 'ict', 'vpa']
                
                for param in key_params:
                    if param in feature_params:
                        logger.info(f"Feature parameter '{param}' found: {feature_params[param]}")
                    else:
                        result['issues'].append(f"Missing feature parameter: {param}")
                        
            else:
                result['success'] = False
                result['issues'].append("Base configuration not loaded properly")
                
        except Exception as e:
            result['success'] = False
            result['issues'].append(f"Error testing config loading: {str(e)}")
            
        return result
    
    def test_config_propagation_to_modules(self) -> Dict[str, Any]:
        """Test how configuration is passed to individual feature modules."""
        logger.info("Testing configuration propagation to feature modules")
        
        result = {
            'success': True,
            'module_tests': {},
            'issues': []
        }
        
        try:
            df = self.create_sample_data(50)
            
            # Test TA module configuration
            logger.info("Testing TA module configuration")
            ta_config = self.base_config.get('features', {}).get('ta', {})
            logger.info(f"TA config received: {ta_config}")
            
            # Test with default config
            ta_features_default = compute_ta_features(df, ta_config)
            logger.info(f"TA features shape with default config: {ta_features_default.shape}")
            
            # Test with modified config
            modified_ta_config = ta_config.copy()
            modified_ta_config['atr_window'] = 10
            modified_ta_config['rvol_windows'] = [10, 30]
            
            ta_features_modified = compute_ta_features(df, modified_ta_config)
            logger.info(f"TA features shape with modified config: {ta_features_modified.shape}")
            
            # Check if configuration change had effect
            ta_columns_default = [col for col in ta_features_default.columns if col.startswith('atr_') or col.startswith('rvol_')]
            ta_columns_modified = [col for col in ta_features_modified.columns if col.startswith('atr_') or col.startswith('rvol_')]
            
            result['module_tests']['ta_basic'] = {
                'default_columns': ta_columns_default,
                'modified_columns': ta_columns_modified,
                'config_change_effect': ta_columns_default != ta_columns_modified
            }
            
            # Test Volume Profile module configuration
            logger.info("Testing Volume Profile module configuration")
            vp_config = self.base_config.get('features', {}).get('volume_profile', {})
            logger.info(f"Volume Profile config received: {vp_config}")
            
            vp_features_default = compute_volume_profile_features(df, vp_config)
            logger.info(f"Volume Profile features shape: {vp_features_default.shape}")
            
            # Test with modified config
            modified_vp_config = vp_config.copy()
            modified_vp_config['bin_size'] = 0.1  # Change from 0.05
            
            vp_features_modified = compute_volume_profile_features(df, modified_vp_config)
            logger.info(f"Volume Profile features shape with modified config: {vp_features_modified.shape}")
            
            result['module_tests']['volume_profile'] = {
                'default_bin_size': vp_config.get('bin_size', 'not_found'),
                'modified_bin_size': modified_vp_config.get('bin_size', 'not_found'),
                'config_change_effect': vp_features_default.shape != vp_features_modified.shape
            }
            
            # Test ICT module configuration
            logger.info("Testing ICT module configuration")
            ict_config = self.base_config.get('features', {}).get('ict', {})
            logger.info(f"ICT config received: {ict_config}")
            
            # ICT features depend on TA features, so create working_df
            ta_features = compute_ta_features(df, self.base_config.get('features', {}).get('ta', {}))
            working_df = df.copy()
            ta_cols = [col for col in ta_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            for col in ta_cols:
                working_df[col] = ta_features[col]
            
            ict_features_default = compute_ict_features(working_df, ict_config)
            logger.info(f"ICT features shape: {ict_features_default.shape}")
            
            # Test with modified config
            modified_ict_config = ict_config.copy()
            modified_ict_config['fvg_min_size_atr'] = 0.5  # Change from 0.25
            
            ict_features_modified = compute_ict_features(working_df, modified_ict_config)
            logger.info(f"ICT features shape with modified config: {ict_features_modified.shape}")
            
            # Check if FVG detection changed
            fvg_up_default = ict_features_default['ict_fvg_up'].sum()
            fvg_up_modified = ict_features_modified['ict_fvg_up'].sum()
            
            result['module_tests']['ict'] = {
                'default_fvg_min_size': ict_config.get('fvg_min_size_atr', 'not_found'),
                'modified_fvg_min_size': modified_ict_config.get('fvg_min_size_atr', 'not_found'),
                'default_fvg_up_count': fvg_up_default,
                'modified_fvg_up_count': fvg_up_modified,
                'config_change_effect': fvg_up_default != fvg_up_modified
            }
            
            # Test VPA module configuration
            logger.info("Testing VPA module configuration")
            vpa_config = self.base_config.get('features', {}).get('vpa', {})
            logger.info(f"VPA config received: {vpa_config}")
            
            vpa_features_default = compute_vpa_features(working_df, vpa_config)
            logger.info(f"VPA features shape: {vpa_features_default.shape}")
            
            # Test with modified config
            modified_vpa_config = vpa_config.copy()
            modified_vpa_config['rvol_climax'] = 3.0  # Change from 2.5
            
            vpa_features_modified = compute_vpa_features(working_df, modified_vpa_config)
            logger.info(f"VPA features shape with modified config: {vpa_features_modified.shape}")
            
            # Check if climax detection changed
            climax_up_default = vpa_features_default['vpa_climax_up'].sum()
            climax_up_modified = vpa_features_modified['vpa_climax_up'].sum()
            
            result['module_tests']['vpa'] = {
                'default_rvol_climax': vpa_config.get('rvol_climax', 'not_found'),
                'modified_rvol_climax': modified_vpa_config.get('rvol_climax', 'not_found'),
                'default_climax_up_count': climax_up_default,
                'modified_climax_up_count': climax_up_modified,
                'config_change_effect': climax_up_default != climax_up_modified
            }
            
        except Exception as e:
            result['success'] = False
            result['issues'].append(f"Error testing config propagation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        return result
    
    def test_feature_union_config_routing(self) -> Dict[str, Any]:
        """Test how feature_union.py routes configuration to individual modules."""
        logger.info("Testing feature_union configuration routing")
        
        result = {
            'success': True,
            'config_routing': {},
            'issues': []
        }
        
        try:
            df = self.create_sample_data(50)
            
            # Test with default configuration
            logger.info("Testing feature_union with default configuration")
            feature_matrix_default = build_feature_matrix(df, self.base_config)
            logger.info(f"Feature matrix shape with default config: {feature_matrix_default.shape}")
            
            # Test with modified configuration
            logger.info("Testing feature_union with modified configuration")
            modified_config = self.base_config.copy()
            
            # Modify key parameters
            modified_config['features']['atr_window'] = 10
            modified_config['features']['rvol_windows'] = [10, 30]
            modified_config['features']['volume_profile']['bin_size'] = 0.1
            modified_config['features']['ict']['fvg_min_size_atr'] = 0.5
            modified_config['features']['vpa']['rvol_climax'] = 3.0
            
            feature_matrix_modified = build_feature_matrix(df, modified_config)
            logger.info(f"Feature matrix shape with modified config: {feature_matrix_modified.shape}")
            
            # Compare results
            default_columns = set(feature_matrix_default.columns)
            modified_columns = set(feature_matrix_modified.columns)
            
            result['config_routing']['column_difference'] = list(default_columns.symmetric_difference(modified_columns))
            result['config_routing']['shape_change'] = feature_matrix_default.shape != feature_matrix_modified.shape
            
            # Check specific feature columns that should change
            ta_columns_default = [col for col in feature_matrix_default.columns if col.startswith('atr_') or col.startswith('rvol_')]
            ta_columns_modified = [col for col in feature_matrix_modified.columns if col.startswith('atr_') or col.startswith('rvol_')]
            
            result['config_routing']['ta_columns_changed'] = ta_columns_default != ta_columns_modified
            result['config_routing']['ta_columns_default'] = ta_columns_default
            result['config_routing']['ta_columns_modified'] = ta_columns_modified
            
            # Check if feature values changed (for binary features like ICT/VPA)
            ict_fvg_up_default = feature_matrix_default['ict_fvg_up'].sum() if 'ict_fvg_up' in feature_matrix_default.columns else 0
            ict_fvg_up_modified = feature_matrix_modified['ict_fvg_up'].sum() if 'ict_fvg_up' in feature_matrix_modified.columns else 0
            
            vpa_climax_up_default = feature_matrix_default['vpa_climax_up'].sum() if 'vpa_climax_up' in feature_matrix_default.columns else 0
            vpa_climax_up_modified = feature_matrix_modified['vpa_climax_up'].sum() if 'vpa_climax_up' in feature_matrix_modified.columns else 0
            
            result['config_routing']['ict_fvg_up_changed'] = ict_fvg_up_default != ict_fvg_up_modified
            result['config_routing']['vpa_climax_up_changed'] = vpa_climax_up_default != vpa_climax_up_modified
            
            result['config_routing']['ict_fvg_up_default'] = ict_fvg_up_default
            result['config_routing']['ict_fvg_up_modified'] = ict_fvg_up_modified
            result['config_routing']['vpa_climax_up_default'] = vpa_climax_up_default
            result['config_routing']['vpa_climax_up_modified'] = vpa_climax_up_modified
            
        except Exception as e:
            result['success'] = False
            result['issues'].append(f"Error testing feature_union config routing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        return result
    
    def analyze_config_flow(self) -> Dict[str, Any]:
        """Analyze the complete configuration flow from loading to usage."""
        logger.info("Analyzing complete configuration flow")
        
        results = {
            'config_loading': self.test_config_loading(),
            'config_propagation': self.test_config_propagation_to_modules(),
            'feature_union_routing': self.test_feature_union_config_routing()
        }
        
        return results
    
    def generate_diagnosis_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive diagnosis report."""
        logger.info("Generating diagnosis report")
        
        report = []
        report.append("=" * 80)
        report.append("CONFIGURATION INTEGRATION DIAGNOSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Config Loading Analysis
        loading_result = results.get('config_loading', {})
        report.append("1. CONFIGURATION LOADING ANALYSIS")
        report.append("-" * 40)
        
        if loading_result.get('success'):
            report.append("✅ Base configuration loaded successfully")
            report.append(f"   - Config structure: {loading_result.get('config_structure', {})}")
            
            if loading_result.get('issues'):
                report.append("⚠️  Issues found:")
                for issue in loading_result['issues']:
                    report.append(f"     - {issue}")
            else:
                report.append("✅ No issues found in configuration loading")
        else:
            report.append("❌ Configuration loading failed")
            for issue in loading_result.get('issues', []):
                report.append(f"   - {issue}")
        
        report.append("")
        
        # Config Propagation Analysis
        propagation_result = results.get('config_propagation', {})
        report.append("2. CONFIGURATION PROPAGATION ANALYSIS")
        report.append("-" * 40)
        
        if propagation_result.get('success'):
            module_tests = propagation_result.get('module_tests', {})
            
            for module_name, test_result in module_tests.items():
                report.append(f"Module: {module_name}")
                
                if module_name == 'ta_basic':
                    report.append(f"   - Default columns: {test_result.get('default_columns', [])}")
                    report.append(f"   - Modified columns: {test_result.get('modified_columns', [])}")
                    report.append(f"   - Config change effect: {'✅ YES' if test_result.get('config_change_effect') else '❌ NO'}")
                
                elif module_name == 'volume_profile':
                    report.append(f"   - Default bin_size: {test_result.get('default_bin_size', 'not_found')}")
                    report.append(f"   - Modified bin_size: {test_result.get('modified_bin_size', 'not_found')}")
                    report.append(f"   - Config change effect: {'✅ YES' if test_result.get('config_change_effect') else '❌ NO'}")
                
                elif module_name == 'ict':
                    report.append(f"   - Default fvg_min_size_atr: {test_result.get('default_fvg_min_size', 'not_found')}")
                    report.append(f"   - Modified fvg_min_size_atr: {test_result.get('modified_fvg_min_size', 'not_found')}")
                    report.append(f"   - Default FVG up count: {test_result.get('default_fvg_up_count', 0)}")
                    report.append(f"   - Modified FVG up count: {test_result.get('modified_fvg_up_count', 0)}")
                    report.append(f"   - Config change effect: {'✅ YES' if test_result.get('config_change_effect') else '❌ NO'}")
                
                elif module_name == 'vpa':
                    report.append(f"   - Default rvol_climax: {test_result.get('default_rvol_climax', 'not_found')}")
                    report.append(f"   - Modified rvol_climax: {test_result.get('modified_rvol_climax', 'not_found')}")
                    report.append(f"   - Default climax up count: {test_result.get('default_climax_up_count', 0)}")
                    report.append(f"   - Modified climax up count: {test_result.get('modified_climax_up_count', 0)}")
                    report.append(f"   - Config change effect: {'✅ YES' if test_result.get('config_change_effect') else '❌ NO'}")
                
                report.append("")
            
            if propagation_result.get('issues'):
                report.append("⚠️  Issues found:")
                for issue in propagation_result['issues']:
                    report.append(f"   - {issue}")
        else:
            report.append("❌ Configuration propagation testing failed")
            for issue in propagation_result.get('issues', []):
                report.append(f"   - {issue}")
        
        report.append("")
        
        # Feature Union Routing Analysis
        routing_result = results.get('feature_union_routing', {})
        report.append("3. FEATURE UNION CONFIGURATION ROUTING ANALYSIS")
        report.append("-" * 40)
        
        if routing_result.get('success'):
            routing = routing_result.get('config_routing', {})
            
            report.append(f"   - Shape change: {'✅ YES' if routing.get('shape_change') else '❌ NO'}")
            report.append(f"   - TA columns changed: {'✅ YES' if routing.get('ta_columns_changed') else '❌ NO'}")
            report.append(f"   - ICT FVG up changed: {'✅ YES' if routing.get('ict_fvg_up_changed') else '❌ NO'}")
            report.append(f"   - VPA climax up changed: {'✅ YES' if routing.get('vpa_climax_up_changed') else '❌ NO'}")
            
            report.append("")
            report.append("   Detailed results:")
            report.append(f"   - Default TA columns: {routing.get('ta_columns_default', [])}")
            report.append(f"   - Modified TA columns: {routing.get('ta_columns_modified', [])}")
            report.append(f"   - Default ICT FVG up count: {routing.get('ict_fvg_up_default', 0)}")
            report.append(f"   - Modified ICT FVG up count: {routing.get('ict_fvg_up_modified', 0)}")
            report.append(f"   - Default VPA climax up count: {routing.get('vpa_climax_up_default', 0)}")
            report.append(f"   - Modified VPA climax up count: {routing.get('vpa_climax_up_modified', 0)}")
            
            column_diff = routing.get('column_difference', [])
            if column_diff:
                report.append(f"   - Column differences: {column_diff}")
            
        else:
            report.append("❌ Feature union routing testing failed")
            for issue in routing_result.get('issues', []):
                report.append(f"   - {issue}")
        
        report.append("")
        
        # Root Cause Analysis
        report.append("4. ROOT CAUSE ANALYSIS")
        report.append("-" * 40)
        
        # Analyze the results to identify root causes
        working_modules = []
        non_working_modules = []
        
        if propagation_result.get('success'):
            module_tests = propagation_result.get('module_tests', {})
            for module_name, test_result in module_tests.items():
                if test_result.get('config_change_effect'):
                    working_modules.append(module_name)
                else:
                    non_working_modules.append(module_name)
        
        if working_modules:
            report.append(f"✅ Modules working correctly: {working_modules}")
        
        if non_working_modules:
            report.append(f"❌ Modules NOT responding to config changes: {non_working_modules}")
        
        # Check if feature_union is routing correctly
        if routing_result.get('success'):
            routing = routing_result.get('config_routing', {})
            if not routing.get('ta_columns_changed'):
                report.append("❌ Feature union not routing TA config changes correctly")
            if not routing.get('ict_fvg_up_changed'):
                report.append("❌ Feature union not routing ICT config changes correctly")
            if not routing.get('vpa_climax_up_changed'):
                report.append("❌ Feature union not routing VPA config changes correctly")
        
        # Overall assessment
        report.append("")
        report.append("5. OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        if non_working_modules or not routing_result.get('success'):
            report.append("❌ CONFIGURATION INTEGRATION FAILURE CONFIRMED")
            report.append("")
            report.append("Likely root causes:")
            if non_working_modules:
                report.append(f"   - Individual modules {non_working_modules} not using config parameters")
            if routing_result.get('success') and not routing.get('config_routing', {}).get('ta_columns_changed'):
                report.append("   - Feature union not extracting/passing config correctly")
            report.append("   - Configuration parameters not being propagated through the system")
        else:
            report.append("✅ Configuration integration appears to be working correctly")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function to run configuration integration diagnosis."""
    print("Starting configuration integration diagnosis...")
    
    tester = ConfigIntegrationTester()
    results = tester.analyze_config_flow()
    
    # Generate and print report
    report = tester.generate_diagnosis_report(results)
    print(report)
    
    # Save report to file
    with open('config_diagnosis_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nDiagnosis report saved to: config_diagnosis_report.txt")
    
    return 0

if __name__ == "__main__":
    exit(main())