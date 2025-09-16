#!/usr/bin/env python3
"""
Comprehensive integration test script for M2 (Features v1) feature modules.

This script tests:
1. Complete feature pipeline through build_feature_matrix()
2. Individual feature modules
3. Data alignment and integration
4. Configuration integration
5. Edge cases and performance
"""

import pandas as pd
import numpy as np
import yaml
import logging
import time
import traceback
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
from volume_price_trade.features.time_of_day import compute_time_of_day_features
from volume_price_trade.features.vwap import compute_vwap_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationTester:
    """Comprehensive integration tester for feature modules."""
    
    def __init__(self):
        self.config = self._load_config()
        self.test_results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        try:
            config_path = Path("config/base.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            features_path = Path("config/features.yaml")
            with open(features_path, "r") as f:
                features_config = yaml.safe_load(f)
            
            # Merge configurations
            config.update(features_config)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def create_sample_ohlcv_data(self, n_bars: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        logger.info(f"Creating sample OHLCV data with {n_bars} bars")
        
        # Create datetime index (trading days only)
        start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')  # Start on a trading day
        dates = pd.date_range(start=start_date, periods=n_bars, freq='5min')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 100.0
        
        # Generate price movements with some autocorrelation
        returns = np.random.normal(0, 0.001, n_bars)
        prices = [base_price]
        
        for i in range(1, n_bars):
            # Add some momentum
            momentum = 0.1 * returns[i-1] if i > 0 else 0
            price_change = returns[i] + momentum
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
        logger.info(f"Created sample data with shape: {df.shape}")
        return df
    
    def create_edge_case_data(self) -> Dict[str, pd.DataFrame]:
        """Create various edge case datasets for testing."""
        logger.info("Creating edge case datasets")
        
        edge_cases = {}
        
        # 1. Very small dataset
        edge_cases['small'] = self.create_sample_ohlcv_data(5)
        
        # 2. Dataset with gaps
        df = self.create_sample_ohlcv_data(50)
        # Remove some rows to create gaps
        df = df.drop(df.index[10:15])
        df = df.drop(df.index[30:35])
        edge_cases['gaps'] = df
        
        # 3. Dataset with zero volume
        df = self.create_sample_ohlcv_data(30)
        df['volume'] = 0
        edge_cases['zero_volume'] = df
        
        # 4. Dataset with constant prices
        df = self.create_sample_ohlcv_data(30)
        df['open'] = 100
        df['high'] = 100
        df['low'] = 100
        df['close'] = 100
        edge_cases['constant_prices'] = df
        
        # 5. Dataset with extreme values
        df = self.create_sample_ohlcv_data(30)
        df.loc[df.index[10], 'high'] = 1000  # Extreme spike
        df.loc[df.index[20], 'low'] = 10    # Extreme drop
        df.loc[df.index[15], 'volume'] = 1000000  # Extreme volume
        edge_cases['extreme_values'] = df
        
        return edge_cases
    
    def test_complete_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test the complete feature pipeline."""
        logger.info("Testing complete feature pipeline")
        
        result = {
            'success': False,
            'error': None,
            'feature_count': 0,
            'expected_features': 68,
            'shape': None,
            'execution_time': None,
            'nan_count': 0,
            'future_leakage': False
        }
        
        try:
            start_time = time.time()
            
            # Build feature matrix
            feature_matrix = build_feature_matrix(df, self.config)
            
            execution_time = time.time() - start_time
            
            # Check results
            result['shape'] = feature_matrix.shape
            result['feature_count'] = len(feature_matrix.columns)
            result['execution_time'] = execution_time
            
            # Count NaN values
            result['nan_count'] = feature_matrix.isna().sum().sum()
            
            # Check for future leakage
            result['future_leakage'] = self._check_future_leakage(feature_matrix)
            
            # Check if we got the expected number of features
            if result['feature_count'] >= result['expected_features']:
                result['success'] = True
                logger.info(f"Pipeline test successful: {result['feature_count']} features generated")
            else:
                logger.warning(f"Pipeline test: Expected {result['expected_features']} features, got {result['feature_count']}")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Pipeline test failed: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def test_individual_modules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test each feature module independently."""
        logger.info("Testing individual feature modules")
        
        modules = {
            'ta_basic': compute_ta_features,
            'volume_profile': compute_volume_profile_features,
            'vpa': compute_vpa_features,
            'ict': compute_ict_features,
            'time_of_day': compute_time_of_day_features,
            'vwap': compute_vwap_features
        }
        
        results = {}
        
        for module_name, module_func in modules.items():
            logger.info(f"Testing {module_name} module")
            
            result = {
                'success': False,
                'error': None,
                'feature_count': 0,
                'shape': None,
                'execution_time': None,
                'nan_count': 0
            }
            
            try:
                start_time = time.time()
                
                # Get module-specific config
                module_config = self.config.get(module_name, {})
                
                # Compute features
                if module_name in ['vpa', 'ict']:
                    # These modules depend on TA features
                    ta_features = compute_ta_features(df, self.config.get('ta', {}))
                    working_df = df.copy()
                    ta_cols = [col for col in ta_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                    for col in ta_cols:
                        working_df[col] = ta_features[col]
                    features = module_func(working_df, module_config)
                else:
                    features = module_func(df, module_config)
                
                execution_time = time.time() - start_time
                
                # Check results
                result['shape'] = features.shape
                result['feature_count'] = len([col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
                result['execution_time'] = execution_time
                result['nan_count'] = features.isna().sum().sum()
                
                result['success'] = True
                logger.info(f"{module_name} test successful: {result['feature_count']} features generated")
                
            except Exception as e:
                result['error'] = str(e)
                result['traceback'] = traceback.format_exc()
                logger.error(f"{module_name} test failed: {e}")
                logger.error(traceback.format_exc())
            
            results[module_name] = result
        
        return results
    
    def test_data_alignment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test data alignment and integration."""
        logger.info("Testing data alignment and integration")
        
        result = {
            'success': False,
            'error': None,
            'alignment_issues': [],
            'column_order_stable': False,
            'missing_data_handled': False
        }
        
        try:
            # Build feature matrix
            feature_matrix = build_feature_matrix(df, self.config)
            
            # Check if index is preserved
            if not feature_matrix.index.equals(df.index):
                result['alignment_issues'].append("Index not preserved")
            
            # Check if original OHLCV columns are present
            original_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_ohlcv = [col for col in original_cols if col not in feature_matrix.columns]
            if missing_ohlcv:
                result['alignment_issues'].append(f"Missing OHLCV columns: {missing_ohlcv}")
            
            # Check column order stability
            expected_order = [
                'open', 'high', 'low', 'close', 'volume',
                'true_range', 'atr_20', 'rvol_5', 'rvol_20',
                'log_return', 'pct_return',
                'close_mean_5', 'close_std_5', 'close_min_5', 'close_max_5',
                'close_mean_10', 'close_std_10', 'close_min_10', 'close_max_10',
                'close_mean_20', 'close_std_20', 'close_min_20', 'close_max_20',
                'volume_mean_5', 'volume_std_5', 'volume_min_5', 'volume_max_5',
                'volume_mean_10', 'volume_std_10', 'volume_min_10', 'volume_max_10',
                'volume_mean_20', 'volume_std_20', 'volume_min_20', 'volume_max_20',
                'vp_poc', 'vp_vah', 'vp_val', 'vp_dist_to_poc_atr',
                'vp_inside_value', 'vp_hvn_near', 'vp_lvn_near', 'vp_poc_shift_dir',
                'vpa_climax_up', 'vpa_climax_down', 'vpa_vdu',
                'vpa_churn', 'vpa_effort_no_result', 'vpa_breakout_conf',
                'ict_fvg_up', 'ict_fvg_down', 'ict_liquidity_sweep_up',
                'ict_liquidity_sweep_down', 'ict_displacement_up', 'ict_displacement_down',
                'ict_dist_to_eq', 'ict_killzone_ny_open', 'ict_killzone_lunch',
                'ict_killzone_pm_drive',
                'minute_of_day', 'sin_minute', 'cos_minute',
                'hour_of_day', 'minute_of_hour', 'is_rth',
                'time_to_close', 'time_since_open', 'is_ny_open',
                'is_lunch', 'is_pm_drive',
                'vwap_session', 'vwap_rolling_20', 'dist_close_to_vwap_session_atr',
                'above_vwap_session', 'vwap_cross_up', 'vwap_cross_down'
            ]
            
            # Check if columns are in expected order (at least the ones that exist)
            existing_cols = [col for col in expected_order if col in feature_matrix.columns]
            actual_cols = [col for col in feature_matrix.columns if col in expected_order]
            
            if existing_cols == actual_cols:
                result['column_order_stable'] = True
            else:
                result['alignment_issues'].append("Column order not stable")
            
            # Check if missing data is handled
            if feature_matrix.isna().sum().sum() < len(feature_matrix) * len(feature_matrix.columns) * 0.5:
                result['missing_data_handled'] = True
            
            if not result['alignment_issues']:
                result['success'] = True
                logger.info("Data alignment test successful")
            else:
                logger.warning(f"Data alignment issues: {result['alignment_issues']}")
                
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Data alignment test failed: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def _check_future_leakage(self, df: pd.DataFrame) -> bool:
        """Check for future leakage in feature calculations."""
        # Check for NaN values at the end of series (which might indicate future leakage)
        for col in df.columns:
            # Skip original OHLCV columns
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue
                
            col_data = df[col]
            
            # Check if there are NaN values at the end
            if col_data.isna().any():
                last_valid_idx = col_data.last_valid_index()
                if last_valid_idx != df.index[-1]:
                    logger.warning(f"Column {col} has NaN values at the end, possible future leakage")
                    return True
        
        return False
    
    def test_configuration_integration(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test configuration integration."""
        logger.info("Testing configuration integration")
        
        result = {
            'success': False,
            'error': None,
            'config_tests': []
        }
        
        try:
            # Test 1: Default configuration
            logger.info("Testing default configuration")
            feature_matrix_default = build_feature_matrix(df, self.config)
            default_shape = feature_matrix_default.shape
            
            # Test 2: Modified configuration
            logger.info("Testing modified configuration")
            modified_config = self.config.copy()
            modified_config['atr_window'] = 10  # Change ATR window
            modified_config['rvol_windows'] = [10, 30]  # Change RVOL windows
            
            feature_matrix_modified = build_feature_matrix(df, modified_config)
            modified_shape = feature_matrix_modified.shape
            
            # Check if configuration changes had effect
            config_test = {
                'name': 'Configuration changes',
                'success': default_shape != modified_shape,
                'details': f"Default: {default_shape}, Modified: {modified_shape}"
            }
            result['config_tests'].append(config_test)
            
            # Test 3: Missing configuration parameters
            logger.info("Testing missing configuration parameters")
            minimal_config = {}  # Empty config should use defaults
            
            feature_matrix_minimal = build_feature_matrix(df, minimal_config)
            minimal_shape = feature_matrix_minimal.shape
            
            config_test = {
                'name': 'Missing config parameters',
                'success': minimal_shape[1] > 5,  # Should still generate some features
                'details': f"Minimal config shape: {minimal_shape}"
            }
            result['config_tests'].append(config_test)
            
            # Check if all tests passed
            if all(test['success'] for test in result['config_tests']):
                result['success'] = True
                logger.info("Configuration integration test successful")
            else:
                logger.warning("Some configuration tests failed")
                
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Configuration integration test failed: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test with edge cases."""
        logger.info("Testing edge cases")
        
        edge_cases = self.create_edge_case_data()
        results = {}
        
        for case_name, df in edge_cases.items():
            logger.info(f"Testing edge case: {case_name}")
            
            result = {
                'success': False,
                'error': None,
                'feature_count': 0,
                'nan_count': 0,
                'warnings': []
            }
            
            try:
                feature_matrix = build_feature_matrix(df, self.config)
                
                result['feature_count'] = len(feature_matrix.columns)
                result['nan_count'] = feature_matrix.isna().sum().sum()
                
                # Check for warnings
                if result['nan_count'] > len(feature_matrix) * len(feature_matrix.columns) * 0.8:
                    result['warnings'].append("High percentage of NaN values")
                
                if len(feature_matrix) < 5:
                    result['warnings'].append("Very few rows in output")
                
                result['success'] = True
                logger.info(f"Edge case {case_name} test successful")
                
            except Exception as e:
                result['error'] = str(e)
                result['traceback'] = traceback.format_exc()
                logger.error(f"Edge case {case_name} test failed: {e}")
            
            results[case_name] = result
        
        return results
    
    def run_performance_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test performance with larger datasets."""
        logger.info("Running performance test")
        
        result = {
            'success': False,
            'error': None,
            'execution_times': {},
            'memory_usage': {}
        }
        
        try:
            # Test with different dataset sizes
            sizes = [100, 500, 1000]
            
            for size in sizes:
                logger.info(f"Testing with {size} bars")
                
                test_df = self.create_sample_ohlcv_data(size)
                
                # Measure execution time
                start_time = time.time()
                feature_matrix = build_feature_matrix(test_df, self.config)
                execution_time = time.time() - start_time
                
                result['execution_times'][size] = execution_time
                
                # Estimate memory usage (rough estimate)
                memory_mb = feature_matrix.memory_usage(deep=True).sum() / 1024 / 1024
                result['memory_usage'][size] = memory_mb
                
                logger.info(f"Size {size}: {execution_time:.2f}s, {memory_mb:.2f}MB")
            
            result['success'] = True
            logger.info("Performance test completed")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Performance test failed: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def run_existing_tests(self) -> Dict[str, Any]:
        """Run existing unit tests."""
        logger.info("Running existing unit tests")
        
        result = {
            'success': False,
            'error': None,
            'test_results': {}
        }
        
        try:
            import pytest
            import subprocess
            
            # Run pytest
            cmd = ['python', '-m', 'pytest', 'tests/', '-v']
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            # Parse results
            output = stdout.decode('utf-8')
            error_output = stderr.decode('utf-8')
            
            result['test_results']['output'] = output
            result['test_results']['error'] = error_output
            result['test_results']['return_code'] = process.returncode
            
            if process.returncode == 0:
                result['success'] = True
                logger.info("Existing tests passed")
            else:
                logger.warning("Some existing tests failed")
                
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Existing tests failed: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting comprehensive integration tests")
        
        # Create test data
        test_df = self.create_sample_ohlcv_data(200)
        
        # Run all tests
        self.test_results['complete_pipeline'] = self.test_complete_pipeline(test_df)
        self.test_results['individual_modules'] = self.test_individual_modules(test_df)
        self.test_results['data_alignment'] = self.test_data_alignment(test_df)
        self.test_results['configuration_integration'] = self.test_configuration_integration(test_df)
        self.test_results['edge_cases'] = self.test_edge_cases()
        self.test_results['performance'] = self.run_performance_test(test_df)
        self.test_results['existing_tests'] = self.run_existing_tests()
        
        return self.test_results
    
    def generate_summary(self) -> str:
        """Generate a comprehensive summary of test results."""
        logger.info("Generating test summary")
        
        summary = []
        summary.append("=" * 80)
        summary.append("COMPREHENSIVE INTEGRATION TEST SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Overall status
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if isinstance(result, dict) and result.get('success', False))
        
        summary.append(f"Overall Status: {passed_tests}/{total_tests} test categories passed")
        summary.append("")
        
        # Complete pipeline
        pipeline_result = self.test_results.get('complete_pipeline', {})
        if pipeline_result.get('success'):
            summary.append("✅ Complete Feature Pipeline: PASSED")
            summary.append(f"   - Features generated: {pipeline_result.get('feature_count', 0)}")
            summary.append(f"   - Expected features: {pipeline_result.get('expected_features', 0)}")
            summary.append(f"   - Execution time: {pipeline_result.get('execution_time', 0):.2f}s")
            summary.append(f"   - NaN count: {pipeline_result.get('nan_count', 0)}")
            summary.append(f"   - Future leakage: {'No' if not pipeline_result.get('future_leakage') else 'Yes'}")
        else:
            summary.append("❌ Complete Feature Pipeline: FAILED")
            summary.append(f"   - Error: {pipeline_result.get('error', 'Unknown')}")
        
        summary.append("")
        
        # Individual modules
        modules_result = self.test_results.get('individual_modules', {})
        summary.append("Individual Feature Modules:")
        for module_name, result in modules_result.items():
            if result.get('success'):
                summary.append(f"✅ {module_name}: PASSED ({result.get('feature_count', 0)} features)")
            else:
                summary.append(f"❌ {module_name}: FAILED ({result.get('error', 'Unknown')})")
        
        summary.append("")
        
        # Data alignment
        alignment_result = self.test_results.get('data_alignment', {})
        if alignment_result.get('success'):
            summary.append("✅ Data Alignment: PASSED")
            summary.append(f"   - Column order stable: {alignment_result.get('column_order_stable', False)}")
            summary.append(f"   - Missing data handled: {alignment_result.get('missing_data_handled', False)}")
        else:
            summary.append("❌ Data Alignment: FAILED")
            issues = alignment_result.get('alignment_issues', [])
            for issue in issues:
                summary.append(f"   - Issue: {issue}")
        
        summary.append("")
        
        # Configuration integration
        config_result = self.test_results.get('configuration_integration', {})
        if config_result.get('success'):
            summary.append("✅ Configuration Integration: PASSED")
        else:
            summary.append("❌ Configuration Integration: FAILED")
            summary.append(f"   - Error: {config_result.get('error', 'Unknown')}")
        
        summary.append("")
        
        # Edge cases
        edge_cases_result = self.test_results.get('edge_cases', {})
        summary.append("Edge Cases:")
        for case_name, result in edge_cases_result.items():
            if result.get('success'):
                summary.append(f"✅ {case_name}: PASSED")
            else:
                summary.append(f"❌ {case_name}: FAILED ({result.get('error', 'Unknown')})")
        
        summary.append("")
        
        # Performance
        perf_result = self.test_results.get('performance', {})
        if perf_result.get('success'):
            summary.append("✅ Performance Test: PASSED")
            exec_times = perf_result.get('execution_times', {})
            for size, time_taken in exec_times.items():
                summary.append(f"   - {size} bars: {time_taken:.2f}s")
        else:
            summary.append("❌ Performance Test: FAILED")
            summary.append(f"   - Error: {perf_result.get('error', 'Unknown')}")
        
        summary.append("")
        
        # Existing tests
        existing_result = self.test_results.get('existing_tests', {})
        if existing_result.get('success'):
            summary.append("✅ Existing Tests: PASSED")
        else:
            summary.append("❌ Existing Tests: FAILED")
            summary.append(f"   - Return code: {existing_result.get('test_results', {}).get('return_code', 'Unknown')}")
        
        summary.append("")
        
        # Issues and recommendations
        summary.append("ISSUES AND RECOMMENDATIONS:")
        summary.append("-" * 40)
        
        # Check for critical issues
        critical_issues = []
        
        if not pipeline_result.get('success'):
            critical_issues.append("Complete pipeline failure - needs immediate attention")
        
        if pipeline_result.get('future_leakage'):
            critical_issues.append("Future leakage detected - serious data integrity issue")
        
        if pipeline_result.get('nan_count', 0) > 100:
            critical_issues.append("High number of NaN values - may affect model performance")
        
        if critical_issues:
            summary.append("Critical Issues:")
            for issue in critical_issues:
                summary.append(f"   - {issue}")
            summary.append("")
        
        # Recommendations
        summary.append("Recommendations:")
        summary.append("   - Review and fix any failed test categories")
        summary.append("   - Monitor NaN values in production")
        summary.append("   - Consider adding more comprehensive unit tests")
        summary.append("   - Document configuration parameters more clearly")
        summary.append("   - Add performance monitoring for large datasets")
        
        summary.append("")
        summary.append("=" * 80)
        
        return "\n".join(summary)


def main():
    """Main function to run all tests."""
    print("Starting comprehensive integration tests for M2 (Features v1)...")
    
    tester = IntegrationTester()
    results = tester.run_all_tests()
    
    # Generate and print summary
    summary = tester.generate_summary()
    print(summary)
    
    # Save summary to file
    with open('integration_test_summary.txt', 'w') as f:
        f.write(summary)
    
    print(f"\nTest results saved to: integration_test_summary.txt")
    print(f"Detailed logs saved to: integration_test.log")
    
    # Return exit code based on results
    pipeline_success = results.get('complete_pipeline', {}).get('success', False)
    return 0 if pipeline_success else 1


if __name__ == "__main__":
    exit(main())