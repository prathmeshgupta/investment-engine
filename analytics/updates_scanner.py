"""Automatic Updates Scanner for Latest Research and Advances in Factor Investing."""

import asyncio
import aiohttp
from datetime import datetime, timedelta
import json
import re
from typing import List, Dict, Any
import feedparser
import arxiv
from bs4 import BeautifulSoup
import yfinance as yf
from pathlib import Path
import sqlite3
import hashlib

class ResearchUpdatesScanner:
    """Scans for latest advances in portfolio construction, factor investing, and academic research."""
    
    def __init__(self):
        self.sources = {
            'arxiv': 'https://arxiv.org/list/q-fin/recent',
            'ssrn': 'https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=1134',
            'nber': 'https://www.nber.org/papers',
            'repec': 'https://ideas.repec.org/n/',
            'google_scholar': 'https://scholar.google.com/scholar_alerts',
            'factor_research': 'https://www.factorresearch.com/research',
            'aqr': 'https://www.aqr.com/Insights/Research',
            'dimensional': 'https://www.dimensional.com/us-en/insights',
            'robeco': 'https://www.robeco.com/en/insights/quantitative-investing/',
            'msci': 'https://www.msci.com/research-and-insights'
        }
        
        self.keywords = [
            'factor investing', 'factor models', 'fama french', 'carhart',
            'momentum', 'value investing', 'quality factors', 'low volatility',
            'smart beta', 'risk parity', 'black litterman', 'portfolio optimization',
            'machine learning finance', 'deep learning portfolio', 'reinforcement learning trading',
            'alternative risk premia', 'factor timing', 'factor crowding',
            'esg factors', 'climate risk', 'crypto factors', 'behavioral factors',
            'multi-factor models', 'factor rotation', 'dynamic asset allocation'
        ]
        
        self.db_path = Path("data/research_updates.db")
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize database for storing research updates."""
        self.db_path.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                source TEXT,
                url TEXT,
                publication_date DATE,
                keywords TEXT,
                relevance_score REAL,
                hash TEXT UNIQUE,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed BOOLEAN DEFAULT 0,
                implemented BOOLEAN DEFAULT 0,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_name TEXT,
                description TEXT,
                performance_metrics TEXT,
                research_id INTEGER,
                implementation_code TEXT,
                backtest_results TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (research_id) REFERENCES research_updates(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def scan_arxiv(self, max_results=50):
        """Scan arXiv for latest quantitative finance papers."""
        updates = []
        
        search = arxiv.Search(
            query="cat:q-fin.PM OR cat:q-fin.ST OR cat:q-fin.RM",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for result in search.results():
            # Check relevance
            relevance = self.calculate_relevance(
                result.title + " " + result.summary
            )
            
            if relevance > 0.3:
                update = {
                    'title': result.title,
                    'authors': ', '.join([author.name for author in result.authors]),
                    'abstract': result.summary,
                    'source': 'arXiv',
                    'url': result.entry_id,
                    'publication_date': result.published.strftime('%Y-%m-%d'),
                    'keywords': self.extract_keywords(result.summary),
                    'relevance_score': relevance,
                    'hash': hashlib.md5(result.entry_id.encode()).hexdigest()
                }
                updates.append(update)
        
        return updates
    
    async def scan_practitioner_insights(self):
        """Scan practitioner websites for new insights."""
        updates = []
        
        async with aiohttp.ClientSession() as session:
            for source, url in self.sources.items():
                if source in ['aqr', 'dimensional', 'robeco', 'msci']:
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                # Extract articles/papers
                                articles = self.extract_articles(soup, source)
                                updates.extend(articles)
                    except Exception as e:
                        print(f"Error scanning {source}: {e}")
        
        return updates
    
    def extract_articles(self, soup, source):
        """Extract articles from practitioner websites."""
        articles = []
        
        # Custom extraction logic for each source
        if source == 'aqr':
            papers = soup.find_all('div', class_='research-item')
            for paper in papers[:10]:
                title = paper.find('h3')
                if title:
                    articles.append({
                        'title': title.text.strip(),
                        'source': 'AQR Capital',
                        'relevance_score': self.calculate_relevance(title.text)
                    })
        
        return articles
    
    def calculate_relevance(self, text):
        """Calculate relevance score based on keywords."""
        text_lower = text.lower()
        score = 0
        
        for keyword in self.keywords:
            if keyword in text_lower:
                score += 1
        
        # Normalize score
        return min(score / len(self.keywords), 1.0)
    
    def extract_keywords(self, text):
        """Extract relevant keywords from text."""
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return ', '.join(found_keywords)
    
    def save_updates(self, updates):
        """Save updates to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for update in updates:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO research_updates 
                    (title, authors, abstract, source, url, publication_date, 
                     keywords, relevance_score, hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    update.get('title'),
                    update.get('authors'),
                    update.get('abstract'),
                    update.get('source'),
                    update.get('url'),
                    update.get('publication_date'),
                    update.get('keywords'),
                    update.get('relevance_score'),
                    update.get('hash')
                ))
            except sqlite3.IntegrityError:
                # Update already exists
                pass
        
        conn.commit()
        conn.close()
    
    def get_recent_updates(self, days=7):
        """Get recent high-relevance updates."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM research_updates
            WHERE discovered_at > datetime('now', '-{} days')
            AND relevance_score > 0.5
            AND reviewed = 0
            ORDER BY relevance_score DESC
            LIMIT 20
        """.format(days))
        
        columns = [description[0] for description in cursor.description]
        updates = []
        for row in cursor.fetchall():
            updates.append(dict(zip(columns, row)))
        
        conn.close()
        return updates
    
    async def scan_all_sources(self):
        """Scan all sources for updates."""
        print("[*] Scanning for latest research updates...")
        
        # Scan arXiv
        arxiv_updates = await self.scan_arxiv()
        print(f"  Found {len(arxiv_updates)} relevant arXiv papers")
        
        # Scan practitioner sites
        practitioner_updates = await self.scan_practitioner_insights()
        print(f"  Found {len(practitioner_updates)} practitioner insights")
        
        # Combine and save
        all_updates = arxiv_updates + practitioner_updates
        self.save_updates(all_updates)
        
        return all_updates
    
    def generate_implementation_suggestions(self, update):
        """Generate code suggestions for implementing new research."""
        suggestions = []
        
        keywords = update.get('keywords', '').lower()
        
        if 'momentum' in keywords:
            suggestions.append({
                'type': 'factor',
                'name': 'Enhanced Momentum',
                'code_template': '''
def enhanced_momentum_factor(returns, lookback=252, skip=21):
    """Enhanced momentum with crash protection."""
    # Skip most recent month to avoid reversal
    momentum = returns.iloc[:-skip].rolling(lookback-skip).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Add volatility scaling
    vol = returns.rolling(lookback).std()
    scaled_momentum = momentum / vol
    
    return scaled_momentum
                '''
            })
        
        if 'machine learning' in keywords or 'deep learning' in keywords:
            suggestions.append({
                'type': 'ml_model',
                'name': 'ML Factor Predictor',
                'code_template': '''
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def ml_factor_prediction(features, target, test_size=0.2):
    """ML-based factor return prediction."""
    X_train, X_test = features[:-test_size], features[-test_size:]
    y_train, y_test = target[:-test_size], target[-test_size:]
    
    # Ensemble approach
    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    # Average predictions
    predictions = (rf.predict(X_test) + xgb_model.predict(X_test)) / 2
    
    return predictions
                '''
            })
        
        return suggestions


class AutomaticUpdater:
    """Automatically applies updates to the investment engine."""
    
    def __init__(self, scanner):
        self.scanner = scanner
        self.update_log = []
        
    async def check_and_apply_updates(self):
        """Check for updates and apply them automatically."""
        # Get recent updates
        updates = self.scanner.get_recent_updates()
        
        applied = []
        for update in updates:
            if update['relevance_score'] > 0.7:
                # Generate implementation
                suggestions = self.scanner.generate_implementation_suggestions(update)
                
                for suggestion in suggestions:
                    # Apply update (would need proper integration)
                    self.apply_update(suggestion, update)
                    applied.append({
                        'update': update['title'],
                        'type': suggestion['type'],
                        'timestamp': datetime.now()
                    })
        
        return applied
    
    def apply_update(self, suggestion, research_update):
        """Apply an update to the system."""
        # This would integrate with the main system
        # For now, we'll save to the database
        conn = sqlite3.connect(str(self.scanner.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO factor_evolution 
            (factor_name, description, implementation_code, research_id)
            VALUES (?, ?, ?, ?)
        """, (
            suggestion['name'],
            f"Auto-generated from: {research_update['title']}",
            suggestion['code_template'],
            research_update.get('id')
        ))
        
        conn.commit()
        conn.close()
        
        print(f"[OK] Applied update: {suggestion['name']}")


async def main():
    """Run the research scanner."""
    scanner = ResearchUpdatesScanner()
    updater = AutomaticUpdater(scanner)
    
    # Scan for updates
    await scanner.scan_all_sources()
    
    # Check and apply updates
    applied = await updater.check_and_apply_updates()
    
    print(f"\n[OK] Scanning complete. Applied {len(applied)} updates.")
    
    # Show recent high-relevance updates
    recent = scanner.get_recent_updates()
    if recent:
        print("\n[*] Recent high-relevance research:")
        for update in recent[:5]:
            print(f"  - {update['title']}")
            print(f"    Relevance: {update['relevance_score']:.2f}")
            print(f"    Source: {update['source']}")


if __name__ == "__main__":
    asyncio.run(main())
