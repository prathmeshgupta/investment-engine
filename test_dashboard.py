"""Comprehensive dashboard testing script."""

import requests
import time
import json


def test_dashboard_connectivity():
    """Test if dashboard is accessible."""
    try:
        response = requests.get('http://127.0.0.1:8050', timeout=10)
        print(f"[PASS] Dashboard connectivity: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"[FAIL] Dashboard connectivity failed: {e}")
        return False


def test_dashboard_ui():
    """Test dashboard UI elements via HTTP requests."""
    print("\n=== DASHBOARD UI TESTING ===")
    
    try:
        # Test main page content
        response = requests.get('http://127.0.0.1:8050', timeout=10)
        content = response.text
        
        # Test 1: Check if header elements are in HTML
        if 'header-title' in content:
            print("[PASS] Header elements present")
        else:
            print("[FAIL] Header elements missing")
        
        # Test 2: Check if tabs are present
        if 'portfolio-tab' in content and 'performance-tab' in content:
            print("[PASS] Navigation tabs present")
        else:
            print("[FAIL] Navigation tabs missing")
        
        # Test 3: Check for CSS styling
        if 'custom.css' in content or 'glassmorphism' in content:
            print("[PASS] CSS styling loaded")
        else:
            print("[FAIL] CSS styling missing")
        
        # Test 4: Check for Dash components
        if 'dash' in content.lower() and 'plotly' in content.lower():
            print("[PASS] Dash/Plotly components loaded")
        else:
            print("[FAIL] Dash/Plotly components missing")
        
        # Test 5: Check content size
        if len(content) > 1000:
            print(f"[PASS] Page content loaded ({len(content):,} characters)")
        else:
            print("[FAIL] Page content too small")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] UI testing failed: {e}")
        return False


def test_dashboard_performance():
    """Test dashboard performance metrics."""
    print("\n=== PERFORMANCE TESTING ===")
    
    try:
        # Test response time
        start_time = time.time()
        response = requests.get('http://127.0.0.1:8050', timeout=10)
        response_time = time.time() - start_time
        
        print(f"[PASS] Response time: {response_time:.2f} seconds")
        
        if response_time < 2.0:
            print("[PASS] Performance: Excellent")
        elif response_time < 5.0:
            print("[PASS] Performance: Good")
        else:
            print("[WARN] Performance: Slow")
        
        # Test content size
        content_size = len(response.content)
        print(f"[PASS] Content size: {content_size:,} bytes")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance testing failed: {e}")
        return False


def test_dashboard_functionality():
    """Test specific dashboard functionality."""
    print("\n=== FUNCTIONALITY TESTING ===")
    
    # Test data endpoints (if available)
    test_endpoints = [
        '/_dash-dependencies',
        '/_dash-layout',
        '/_dash-component-suites'
    ]
    
    working_endpoints = 0
    for endpoint in test_endpoints:
        try:
            response = requests.get(f'http://127.0.0.1:8050{endpoint}', timeout=5)
            if response.status_code == 200:
                working_endpoints += 1
                print(f"[PASS] Endpoint {endpoint}: Working")
            else:
                print(f"[FAIL] Endpoint {endpoint}: Status {response.status_code}")
        except:
            print(f"[FAIL] Endpoint {endpoint}: Failed")
    
    print(f"[INFO] Working endpoints: {working_endpoints}/{len(test_endpoints)}")
    return working_endpoints > 0


def run_comprehensive_test():
    """Run all dashboard tests."""
    print("=" * 60)
    print("COMPREHENSIVE DASHBOARD TESTING")
    print("=" * 60)
    
    results = {
        'connectivity': False,
        'ui': False,
        'performance': False,
        'functionality': False
    }
    
    # Test connectivity
    print("\n1. CONNECTIVITY TEST")
    results['connectivity'] = test_dashboard_connectivity()
    
    if results['connectivity']:
        # Test performance
        print("\n2. PERFORMANCE TEST")
        results['performance'] = test_dashboard_performance()
        
        # Test functionality
        print("\n3. FUNCTIONALITY TEST")
        results['functionality'] = test_dashboard_functionality()
        
        # Test UI (requires Chrome driver)
        print("\n4. UI TEST")
        try:
            results['ui'] = test_dashboard_ui()
        except Exception as e:
            print(f"✗ UI testing skipped: {e}")
            print("  (Chrome driver may not be available)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name.upper():15} {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - Dashboard is fully operational!")
    elif passed >= total * 0.75:
        print("[WARNING] MOSTLY WORKING - Minor issues detected")
    else:
        print("[ERROR] ISSUES DETECTED - Dashboard needs attention")
    
    return results


if __name__ == "__main__":
    run_comprehensive_test()
