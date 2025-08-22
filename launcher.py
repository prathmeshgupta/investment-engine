"""
One-click launcher for Investment Engine - No setup required!
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path
from auto_config import AutoConfig, get_config, get_cache, get_data_feed

def check_requirements():
    """Check and install requirements if needed."""
    print("[*] Checking requirements...")
    
    # Check if pip packages are installed
    required_packages = ['dash', 'pandas', 'yfinance', 'plotly', 'dash-bootstrap-components']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[*] Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("[OK] Packages installed successfully!")
    else:
        print("[OK] All requirements satisfied!")

def launch_dashboard():
    """Launch the investment dashboard."""
    print("\n[*] Starting Investment Engine Dashboard...")
    print("=" * 50)
    
    # Setup configuration automatically
    auto_config = AutoConfig()
    config = auto_config.setup_everything()
    
    print("\nDashboard Configuration:")
    print("  - URL: http://127.0.0.1:8050")
    print("  - Database: SQLite (auto-configured)")
    print("  - Cache: In-memory (no Redis needed)")
    print("  - Market Data: Yahoo Finance (no API key needed)")
    
    # Use the full dashboard with all features
    print("\n[*] Launching full dashboard with all features...")
    print("[*] Loading advanced features:")
    print("  - Research Tab with academic papers")
    print("  - Portfolio Builder with optimization")
    print("  - AI Chatbot assistant")
    print("  - Strategy Builder and Backtesting")
    print("  - Risk Analytics and Monitoring")
    print("  - Real-time Market Data Integration")
    
    # Open browser automatically
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8050")
    
    # Run the main entry point which uses the full dashboard
    subprocess.run([sys.executable, "main.py"])

def main():
    """Main launcher function."""
    print("""
    ================================================
                INVESTMENT ENGINE                  
              AUTO LAUNCHER - ZERO CONFIG         
          No Manual Setup Required!               
    ================================================
    """)
    
    try:
        # Check and install requirements
        check_requirements()
        
        # Launch dashboard
        launch_dashboard()
        
    except KeyboardInterrupt:
        print("\n\n[*] Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
