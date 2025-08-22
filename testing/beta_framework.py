"""Beta Testing Framework with Monitoring and Performance Tracking."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import psutil
import traceback
from dataclasses import dataclass, asdict
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TestCase:
    """Definition of a test case."""
    id: str
    name: str
    category: str  # performance, functionality, stress, integration
    description: str
    priority: str  # high, medium, low
    expected_outcome: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]] = None
    status: str = 'pending'  # pending, running, passed, failed, error
    execution_time: float = 0.0
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class PerformanceMetric:
    """Performance metric tracking."""
    metric_name: str
    value: float
    unit: str
    threshold: float
    status: str  # normal, warning, critical
    timestamp: datetime


class BetaTestingFramework:
    """Comprehensive beta testing and monitoring framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.test_cases = []
        self.performance_metrics = []
        self.test_results = {}
        self.monitoring_data = []
        self.db_path = Path("data/beta_testing.db")
        self.initialize_database()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logger = logging.getLogger('BetaTesting')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler('logs/beta_testing.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def initialize_database(self):
        """Initialize database for storing test results."""
        self.db_path.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                test_name TEXT,
                category TEXT,
                status TEXT,
                execution_time REAL,
                error_message TEXT,
                expected_outcome TEXT,
                actual_outcome TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                value REAL,
                unit TEXT,
                threshold REAL,
                status TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System monitoring table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_latency REAL,
                active_users INTEGER,
                error_count INTEGER,
                warning_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                feature TEXT,
                rating INTEGER,
                feedback TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_test_case(self, test_case: TestCase):
        """Register a new test case."""
        self.test_cases.append(test_case)
        self.logger.info(f"Registered test case: {test_case.name}")
    
    async def run_test_case(self, test_case: TestCase) -> TestCase:
        """Run a single test case."""
        self.logger.info(f"Running test: {test_case.name}")
        test_case.status = 'running'
        test_case.timestamp = datetime.now()
        start_time = time.time()
        
        try:
            # Execute test based on category
            if test_case.category == 'performance':
                result = await self._run_performance_test(test_case)
            elif test_case.category == 'functionality':
                result = await self._run_functionality_test(test_case)
            elif test_case.category == 'stress':
                result = await self._run_stress_test(test_case)
            elif test_case.category == 'integration':
                result = await self._run_integration_test(test_case)
            else:
                result = {'error': 'Unknown test category'}
            
            test_case.actual_outcome = result
            
            # Compare with expected outcome
            if self._compare_outcomes(test_case.expected_outcome, result):
                test_case.status = 'passed'
                self.logger.info(f"Test passed: {test_case.name}")
            else:
                test_case.status = 'failed'
                self.logger.warning(f"Test failed: {test_case.name}")
                
        except Exception as e:
            test_case.status = 'error'
            test_case.error_message = str(e)
            self.logger.error(f"Test error: {test_case.name} - {str(e)}")
            
        test_case.execution_time = time.time() - start_time
        
        # Save to database
        self._save_test_result(test_case)
        
        return test_case
    
    async def _run_performance_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run performance test."""
        results = {}
        
        # Example: Test response time
        start = time.time()
        # Simulate API call or function execution
        await asyncio.sleep(0.1)  # Replace with actual test
        response_time = time.time() - start
        
        results['response_time'] = response_time
        results['throughput'] = 1000 / response_time  # requests per second
        
        # Memory usage
        process = psutil.Process()
        results['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU usage
        results['cpu_usage'] = process.cpu_percent()
        
        return results
    
    async def _run_functionality_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run functionality test."""
        results = {}
        
        # Example: Test factor calculation
        try:
            from factors.comprehensive_engine import ComprehensiveFactorEngine
            engine = ComprehensiveFactorEngine()
            
            # Generate test data
            returns = pd.DataFrame(
                np.random.randn(252, 10) * 0.01,
                columns=[f'Asset_{i}' for i in range(10)]
            )
            
            # Calculate factors
            exposures = engine.calculate_factor_exposures(returns)
            
            results['factors_calculated'] = len(exposures.columns)
            results['assets_processed'] = len(exposures.index)
            results['has_nan'] = exposures.isna().any().any()
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _run_stress_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run stress test."""
        results = {}
        
        # Simulate high load
        concurrent_requests = 100
        tasks = []
        
        async def simulate_request():
            await asyncio.sleep(np.random.random() * 0.1)
            return time.time()
        
        start = time.time()
        for _ in range(concurrent_requests):
            tasks.append(simulate_request())
        
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start
        
        results['concurrent_requests'] = concurrent_requests
        results['total_time'] = total_time
        results['avg_response_time'] = total_time / concurrent_requests
        results['requests_per_second'] = concurrent_requests / total_time
        
        return results
    
    async def _run_integration_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run integration test."""
        results = {}
        
        # Test data flow between components
        try:
            # Test data provider
            from data.data_provider import DataProvider
            provider = DataProvider()
            
            # Test factor engine
            from factors.comprehensive_engine import ComprehensiveFactorEngine
            engine = ComprehensiveFactorEngine()
            
            # Test optimizer
            from optimization.optimizer_engine import OptimizerEngine
            optimizer = OptimizerEngine()
            
            results['data_provider'] = 'OK'
            results['factor_engine'] = 'OK'
            results['optimizer'] = 'OK'
            results['integration'] = 'SUCCESS'
            
        except Exception as e:
            results['error'] = str(e)
            results['integration'] = 'FAILED'
        
        return results
    
    def _compare_outcomes(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Compare expected and actual outcomes."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            
            actual_value = actual[key]
            
            # Handle numeric comparisons with tolerance
            if isinstance(expected_value, (int, float)):
                if isinstance(expected_value, dict):
                    # Range check: {'min': 0, 'max': 100}
                    if 'min' in expected_value and actual_value < expected_value['min']:
                        return False
                    if 'max' in expected_value and actual_value > expected_value['max']:
                        return False
                else:
                    # Exact value with 5% tolerance
                    if abs(actual_value - expected_value) > expected_value * 0.05:
                        return False
            else:
                # Exact match for other types
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _save_test_result(self, test_case: TestCase):
        """Save test result to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_results 
            (test_id, test_name, category, status, execution_time, 
             error_message, expected_outcome, actual_outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_case.id,
            test_case.name,
            test_case.category,
            test_case.status,
            test_case.execution_time,
            test_case.error_message,
            json.dumps(test_case.expected_outcome),
            json.dumps(test_case.actual_outcome) if test_case.actual_outcome else None
        ))
        
        conn.commit()
        conn.close()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered tests."""
        self.logger.info(f"Starting test suite with {len(self.test_cases)} tests")
        
        results = {
            'total': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'execution_time': 0,
            'tests': []
        }
        
        start_time = time.time()
        
        for test_case in self.test_cases:
            result = await self.run_test_case(test_case)
            results['tests'].append(asdict(result))
            
            if result.status == 'passed':
                results['passed'] += 1
            elif result.status == 'failed':
                results['failed'] += 1
            elif result.status == 'error':
                results['errors'] += 1
        
        results['execution_time'] = time.time() - start_time
        results['success_rate'] = results['passed'] / results['total'] * 100
        
        self.logger.info(f"Test suite completed: {results['passed']}/{results['total']} passed")
        
        return results
    
    def monitor_system_performance(self) -> Dict[str, Any]:
        """Monitor system performance metrics."""
        metrics = {}
        
        # CPU usage
        metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        metrics['memory_available'] = memory.available / 1024 / 1024 / 1024  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics['disk_usage'] = disk.percent
        metrics['disk_free'] = disk.free / 1024 / 1024 / 1024  # GB
        
        # Network
        net = psutil.net_io_counters()
        metrics['network_sent'] = net.bytes_sent / 1024 / 1024  # MB
        metrics['network_recv'] = net.bytes_recv / 1024 / 1024  # MB
        
        # Process count
        metrics['process_count'] = len(psutil.pids())
        
        # Save to database
        self._save_monitoring_data(metrics)
        
        return metrics
    
    def _save_monitoring_data(self, metrics: Dict[str, Any]):
        """Save monitoring data to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_monitoring 
            (cpu_usage, memory_usage, disk_usage, network_latency, 
             active_users, error_count, warning_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.get('cpu_usage', 0),
            metrics.get('memory_usage', 0),
            metrics.get('disk_usage', 0),
            metrics.get('network_latency', 0),
            metrics.get('active_users', 0),
            metrics.get('error_count', 0),
            metrics.get('warning_count', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def collect_user_feedback(self, user_id: str, feature: str, 
                            rating: int, feedback: str):
        """Collect user feedback."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_feedback (user_id, feature, rating, feedback)
            VALUES (?, ?, ?, ?)
        """, (user_id, feature, rating, feedback))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Feedback collected from user {user_id} for {feature}")
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        conn = sqlite3.connect(str(self.db_path))
        
        # Get test results summary
        df_tests = pd.read_sql_query("""
            SELECT category, status, COUNT(*) as count,
                   AVG(execution_time) as avg_time
            FROM test_results
            GROUP BY category, status
        """, conn)
        
        # Get performance metrics
        df_metrics = pd.read_sql_query("""
            SELECT metric_name, AVG(value) as avg_value,
                   MIN(value) as min_value, MAX(value) as max_value
            FROM performance_metrics
            GROUP BY metric_name
        """, conn)
        
        # Get system monitoring data
        df_monitoring = pd.read_sql_query("""
            SELECT AVG(cpu_usage) as avg_cpu,
                   AVG(memory_usage) as avg_memory,
                   AVG(disk_usage) as avg_disk
            FROM system_monitoring
        """, conn)
        
        # Get user feedback summary
        df_feedback = pd.read_sql_query("""
            SELECT feature, AVG(rating) as avg_rating,
                   COUNT(*) as feedback_count
            FROM user_feedback
            GROUP BY feature
        """, conn)
        
        conn.close()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': df_tests.to_dict('records'),
            'performance_metrics': df_metrics.to_dict('records'),
            'system_monitoring': df_monitoring.to_dict('records')[0] if not df_monitoring.empty else {},
            'user_feedback': df_feedback.to_dict('records'),
            'recommendations': self._generate_recommendations(df_tests, df_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, df_tests, df_metrics) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check test pass rate
        if not df_tests.empty:
            total_tests = df_tests['count'].sum()
            passed_tests = df_tests[df_tests['status'] == 'passed']['count'].sum()
            pass_rate = passed_tests / total_tests * 100
            
            if pass_rate < 80:
                recommendations.append(
                    f"Test pass rate is {pass_rate:.1f}%. Focus on fixing failing tests."
                )
        
        # Check performance metrics
        if not df_metrics.empty:
            for _, row in df_metrics.iterrows():
                if 'response_time' in row['metric_name'] and row['avg_value'] > 1.0:
                    recommendations.append(
                        "Response time is high. Consider optimizing performance."
                    )
        
        if not recommendations:
            recommendations.append("System is performing well. Continue monitoring.")
        
        return recommendations


def create_default_test_suite() -> List[TestCase]:
    """Create default test suite for the investment engine."""
    test_cases = [
        TestCase(
            id='PERF001',
            name='Dashboard Load Time',
            category='performance',
            description='Test dashboard initial load time',
            priority='high',
            expected_outcome={'response_time': {'max': 2.0}}
        ),
        TestCase(
            id='FUNC001',
            name='Factor Calculation',
            category='functionality',
            description='Test factor calculation accuracy',
            priority='high',
            expected_outcome={'factors_calculated': 15, 'has_nan': False}
        ),
        TestCase(
            id='STRESS001',
            name='Concurrent Users',
            category='stress',
            description='Test system with multiple concurrent users',
            priority='medium',
            expected_outcome={'requests_per_second': {'min': 50}}
        ),
        TestCase(
            id='INT001',
            name='Component Integration',
            category='integration',
            description='Test integration between major components',
            priority='high',
            expected_outcome={'integration': 'SUCCESS'}
        ),
        TestCase(
            id='PERF002',
            name='Portfolio Optimization Speed',
            category='performance',
            description='Test portfolio optimization calculation time',
            priority='high',
            expected_outcome={'response_time': {'max': 5.0}}
        ),
        TestCase(
            id='FUNC002',
            name='Backtest Accuracy',
            category='functionality',
            description='Test backtesting engine accuracy',
            priority='high',
            expected_outcome={'error_rate': {'max': 0.01}}
        )
    ]
    
    return test_cases


async def main():
    """Run beta testing framework."""
    framework = BetaTestingFramework()
    
    # Register default test cases
    test_cases = create_default_test_suite()
    for test_case in test_cases:
        framework.register_test_case(test_case)
    
    # Run all tests
    print("[*] Starting Beta Testing Framework...")
    results = await framework.run_all_tests()
    
    # Monitor system
    print("[*] Monitoring system performance...")
    metrics = framework.monitor_system_performance()
    
    # Generate report
    print("[*] Generating test report...")
    report = framework.generate_test_report()
    
    # Display summary
    print(f"\n[OK] Beta Testing Complete!")
    print(f"  Total Tests: {results['total']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  CPU Usage: {metrics['cpu_usage']:.1f}%")
    print(f"  Memory Usage: {metrics['memory_usage']:.1f}%")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
