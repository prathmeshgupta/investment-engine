"""
Enterprise Architecture Framework for Investment Engine
Modular, scalable, and deployable architecture patterns
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time

class ComponentStatus(Enum):
    """Component status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ComponentMetrics:
    """Component performance metrics."""
    requests_processed: int = 0
    errors_encountered: int = 0
    average_response_time: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

class BaseComponent(ABC):
    """Base class for all system components."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.status = ComponentStatus.INITIALIZING
        self.metrics = ComponentMetrics()
        self.logger = self._setup_logging()
        self.dependencies: List[str] = []
        self.health_check_interval = 30  # seconds
        
    def _setup_logging(self) -> logging.Logger:
        """Setup component-specific logging."""
        logger = logging.getLogger(f"component.{self.name}")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(f'logs/{self.name}.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the component."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status and metrics."""
        return {
            'name': self.name,
            'status': self.status.value,
            'metrics': {
                'requests_processed': self.metrics.requests_processed,
                'errors_encountered': self.metrics.errors_encountered,
                'average_response_time': self.metrics.average_response_time,
                'last_activity': self.metrics.last_activity.isoformat(),
                'memory_usage': self.metrics.memory_usage,
                'cpu_usage': self.metrics.cpu_usage
            }
        }
    
    def update_metrics(self, response_time: float = None, error: bool = False):
        """Update component metrics."""
        self.metrics.requests_processed += 1
        if error:
            self.metrics.errors_encountered += 1
        if response_time:
            # Simple moving average
            total_time = self.metrics.average_response_time * (self.metrics.requests_processed - 1)
            self.metrics.average_response_time = (total_time + response_time) / self.metrics.requests_processed
        self.metrics.last_activity = datetime.now()

class ServiceRegistry:
    """Service registry for component discovery and management."""
    
    def __init__(self):
        self.services: Dict[str, BaseComponent] = {}
        self.service_dependencies: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
    
    def register_service(self, service: BaseComponent):
        """Register a service component."""
        with self.lock:
            self.services[service.name] = service
            self.service_dependencies[service.name] = service.dependencies
    
    def unregister_service(self, service_name: str):
        """Unregister a service component."""
        with self.lock:
            if service_name in self.services:
                del self.services[service_name]
            if service_name in self.service_dependencies:
                del self.service_dependencies[service_name]
    
    def get_service(self, service_name: str) -> Optional[BaseComponent]:
        """Get service by name."""
        return self.services.get(service_name)
    
    def get_all_services(self) -> Dict[str, BaseComponent]:
        """Get all registered services."""
        return self.services.copy()
    
    def get_startup_order(self) -> List[str]:
        """Get service startup order based on dependencies."""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            dependencies = self.service_dependencies.get(service_name, [])
            
            for dep in dependencies:
                if dep in self.services:
                    visit(dep)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        for service_name in self.services:
            if service_name not in visited:
                visit(service_name)
        
        return order

class MessageBus:
    """Event-driven message bus for component communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type."""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
    
    def publish(self, event_type: str, data: Any = None):
        """Publish an event."""
        message = {
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now()
        }
        self.message_queue.put(message)
    
    def start(self):
        """Start the message bus worker."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_messages)
        self.worker_thread.start()
    
    def stop(self):
        """Stop the message bus worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _process_messages(self):
        """Process messages from the queue."""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                event_type = message['event_type']
                
                if event_type in self.subscribers:
                    for handler in self.subscribers[event_type]:
                        try:
                            handler(message)
                        except Exception as e:
                            logging.error(f"Error processing message {event_type}: {e}")
                
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in message bus: {e}")

class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self, config_file: str = "config/application.json"):
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        self.watchers: List[Callable] = []
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                self.config_data = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_file} not found, using defaults")
            self.config_data = self._get_default_config()
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            self.config_data = self._get_default_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._notify_watchers(key, value)
    
    def watch(self, callback: Callable):
        """Watch for configuration changes."""
        self.watchers.append(callback)
    
    def _notify_watchers(self, key: str, value: Any):
        """Notify configuration watchers."""
        for watcher in self.watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logging.error(f"Error notifying config watcher: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "investment_engine",
                "pool_size": 10
            },
            "cache": {
                "type": "redis",
                "host": "localhost",
                "port": 6379,
                "ttl": 3600
            },
            "security": {
                "jwt_secret": "your-secret-key",
                "session_timeout": 3600,
                "rate_limit": 100
            },
            "monitoring": {
                "metrics_interval": 60,
                "health_check_interval": 30,
                "log_level": "INFO"
            }
        }

class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self, service_registry: ServiceRegistry, message_bus: MessageBus):
        self.service_registry = service_registry
        self.message_bus = message_bus
        self.monitoring = False
        self.monitor_thread = None
        self.check_interval = 30  # seconds
        
    def start_monitoring(self):
        """Start health monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_health)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_health(self):
        """Monitor system health."""
        while self.monitoring:
            try:
                services = self.service_registry.get_all_services()
                
                for service_name, service in services.items():
                    try:
                        # Run health check asynchronously
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        is_healthy = loop.run_until_complete(service.health_check())
                        loop.close()
                        
                        if not is_healthy:
                            self.message_bus.publish('service.unhealthy', {
                                'service': service_name,
                                'status': service.status.value
                            })
                        
                    except Exception as e:
                        logging.error(f"Health check failed for {service_name}: {e}")
                        self.message_bus.publish('service.error', {
                            'service': service_name,
                            'error': str(e)
                        })
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Error in health monitoring: {e}")
                time.sleep(self.check_interval)

class ApplicationOrchestrator:
    """Main application orchestrator for managing all components."""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.message_bus = MessageBus()
        self.config_manager = ConfigurationManager()
        self.health_monitor = HealthMonitor(self.service_registry, self.message_bus)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup system event handlers."""
        self.message_bus.subscribe('service.unhealthy', self._handle_unhealthy_service)
        self.message_bus.subscribe('service.error', self._handle_service_error)
        self.message_bus.subscribe('system.shutdown', self._handle_shutdown)
    
    def register_component(self, component: BaseComponent):
        """Register a component with the system."""
        self.service_registry.register_service(component)
        logging.info(f"Registered component: {component.name}")
    
    async def start_system(self):
        """Start the entire system."""
        try:
            logging.info("Starting Investment Engine System...")
            
            # Start message bus
            self.message_bus.start()
            
            # Get startup order
            startup_order = self.service_registry.get_startup_order()
            
            # Initialize and start services in order
            for service_name in startup_order:
                service = self.service_registry.get_service(service_name)
                if service:
                    logging.info(f"Initializing service: {service_name}")
                    await service.initialize()
                    
                    logging.info(f"Starting service: {service_name}")
                    await service.start()
            
            # Start health monitoring
            self.health_monitor.start_monitoring()
            
            self.running = True
            logging.info("Investment Engine System started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start system: {e}")
            await self.shutdown_system()
            raise
    
    async def shutdown_system(self):
        """Shutdown the entire system gracefully."""
        logging.info("Shutting down Investment Engine System...")
        
        self.running = False
        
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        # Stop services in reverse order
        startup_order = self.service_registry.get_startup_order()
        for service_name in reversed(startup_order):
            service = self.service_registry.get_service(service_name)
            if service:
                try:
                    logging.info(f"Stopping service: {service_name}")
                    await service.stop()
                except Exception as e:
                    logging.error(f"Error stopping service {service_name}: {e}")
        
        # Stop message bus
        self.message_bus.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logging.info("Investment Engine System shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        services = self.service_registry.get_all_services()
        service_statuses = {}
        
        for name, service in services.items():
            service_statuses[name] = service.get_status()
        
        return {
            'system_running': self.running,
            'total_services': len(services),
            'services': service_statuses,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_unhealthy_service(self, message: Dict[str, Any]):
        """Handle unhealthy service events."""
        service_name = message['data']['service']
        logging.warning(f"Service {service_name} is unhealthy")
        
        # Implement recovery logic here
        # For example: restart service, failover, etc.
    
    def _handle_service_error(self, message: Dict[str, Any]):
        """Handle service error events."""
        service_name = message['data']['service']
        error = message['data']['error']
        logging.error(f"Service {service_name} encountered error: {error}")
        
        # Implement error handling logic here
    
    def _handle_shutdown(self, message: Dict[str, Any]):
        """Handle system shutdown events."""
        asyncio.create_task(self.shutdown_system())

# Singleton instance
orchestrator = ApplicationOrchestrator()
