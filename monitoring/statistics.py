# monitoring/statistics.py  
"""
Simple Statistics Collector
Thu thập thống kê đơn giản cho hệ thống
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

class SimpleStatsCollector:
    """Lightweight statistics collector"""
    
    def __init__(self):
        self.stats = defaultdict(int)
        self.timing_data = deque(maxlen=100)  # Keep last 100 timings
        self.start_time = datetime.now()
    
    def increment(self, key: str, amount: int = 1):
        """Increment a counter"""
        self.stats[key] += amount
    
    def record_timing(self, operation: str, duration: float):
        """Record operation timing"""
        self.timing_data.append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate timing averages
        timing_summary = {}
        if self.timing_data:
            by_operation = defaultdict(list)
            for entry in self.timing_data:
                by_operation[entry['operation']].append(entry['duration'])
            
            for op, durations in by_operation.items():
                timing_summary[f'{op}_avg'] = sum(durations) / len(durations)
                timing_summary[f'{op}_max'] = max(durations)
        
        return {
            'uptime_seconds': uptime,
            'counters': dict(self.stats),
            'timing': timing_summary,
            'total_operations': len(self.timing_data)
        }
    
    def reset(self):
        """Reset all statistics"""
        self.stats.clear()
        self.timing_data.clear()
        self.start_time = datetime.now()

# Factory functions
def create_parking_monitor(spots_config: List[Dict]) -> ParkingMonitor:
    """Create optimized parking monitor"""
    return ParkingMonitor(spots_config)

def create_stats_collector() -> SimpleStatsCollector:
    """Create simple stats collector"""
    return SimpleStatsCollector()

# Example usage
if __name__ == "__main__":
    # Example spot configuration
    spots = [
        {'id': 'A1', 'name': 'Spot A1', 'polygon': [(100, 100), (200, 100), (200, 200), (100, 200)]},
        {'id': 'A2', 'name': 'Spot A2', 'polygon': [(220, 100), (320, 100), (320, 200), (220, 200)]},
        {'id': 'B1', 'name': 'Spot B1', 'polygon': [(100, 220), (200, 220), (200, 320), (100, 320)]}
    ]
    
    # Create monitor
    monitor = create_parking_monitor(spots)
    
    print("🏁 Optimized Parking Monitor")
    print("=" * 40)
    print(f"✅ Spots configured: {len(spots)}")
    
    # Show current status
    status = monitor.get_current_status()
    print(f"📊 Status: {status['occupied_spots']}/{status['total_spots']} occupied")
    
    # Show spot details
    details = monitor.get_spot_details()
    print("\n📍 Spot Details:")
    for spot in details:
        print(f"  {spot['name']}: {'OCCUPIED' if spot['is_occupied'] else 'EMPTY'}")
    
    print(f"\n🔧 Optimizations applied:")
    print(f"   ✓ Removed complex statistics")
    print(f"   ✓ Simplified event structure") 
    print(f"   ✓ Basic vehicle assignment")
    print(f"   ✓ Minimal memory usage")
    print(f"   ✓ Essential functionality only")
    
    print(f"\n📈 Key Features Retained:")
    print(f"   • 1 vehicle = 1 spot assignment")
    print(f"   • State stability checking")
    print(f"   • Event generation")
    print(f"   • Thread-safe operations")
    print(f"   • Occupancy tracking")