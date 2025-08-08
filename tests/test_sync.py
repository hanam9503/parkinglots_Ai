# Patches for main_parking_system.py to improve plate formatting

# 1. Fix confidence display (around line 485 in _process_vehicle_enter)
def format_confidence_display(confidence):
    """Format confidence as percentage properly"""
    if confidence > 1.0:
        # If confidence > 1, it's already in percentage format, just format it
        return f"{confidence:.1f}%"
    else:
        # Convert to percentage and format
        return f"{confidence * 100:.1f}%"

# 2. Enhanced plate text selection and formatting
def get_best_plate_text(ocr_result):
    """Get the best plate text from OCR result"""
    if not ocr_result:
        return "Không nhận diện được biển", 0.0
    
    validated_text = ocr_result.get('validated_text')
    raw_text = ocr_result.get('text')
    confidence = ocr_result.get('confidence', 0.0)
    
    # Prefer validated text if available and valid
    if validated_text and ocr_result.get('is_valid', False):
        # Clean up double line formatting for display
        display_text = validated_text.replace('\n', ' ')  # Replace newline with space for logs
        return display_text, confidence
    elif raw_text:
        # Use raw text but apply basic cleaning
        clean_text = raw_text.replace('\n', ' ')
        return clean_text, confidence
    else:
        return "Không nhận diện được biển", 0.0

# 3. Enhanced logging format for plate events  
def log_plate_event(action, spot_name, plate_text, confidence, sent_status="✅"):
    """Enhanced logging for plate events"""
    
    # Format confidence properly
    conf_display = format_confidence_display(confidence) if confidence > 0 else "0%"
    
    # Clean plate text for display
    if plate_text and plate_text != "Không nhận diện được biển":
        plate_display = plate_text.strip()
    else:
        plate_display = "No plate detected"
    
    if action == 'enter':
        logger.info(f"{sent_status} 🚗 {spot_name}: Vehicle entered - {plate_display} (conf: {conf_display})")
    elif action == 'exit':
        logger.info(f"{sent_status} 🚪 {spot_name}: Vehicle exited - {plate_display} (conf: {conf_display})")

# 4. Patch for _process_vehicle_enter method
"""
Replace this section in _process_vehicle_enter (around line 580-590):

OLD CODE:
                    if plate_result:
                        plate_text = plate_result['validated_text'] or plate_result['text']
                        plate_confidence = plate_result['confidence']
                        plate_image_path = plate_result.get('save_path')
                        
                        self.stats['plates_detected'] += 1
                        if plate_result['is_valid']:
                            self.stats['plates_validated'] += 1
                        
                        logger.info(f"   ✅ Plate detected: {plate_text} (conf: {plate_confidence:.3f})")
                    else:
                        logger.info(f"   ❌ No plate detected")

NEW CODE:
                    if plate_result:
                        plate_text, plate_confidence = get_best_plate_text(plate_result)
                        plate_image_path = plate_result.get('save_path')
                        
                        self.stats['plates_detected'] += 1
                        if plate_result.get('is_valid', False):
                            self.stats['plates_validated'] += 1
                        
                        # Enhanced logging with proper formatting
                        conf_display = format_confidence_display(plate_confidence)
                        validation_status = "✅" if plate_result.get('is_valid', False) else "⚠️"
                        logger.info(f"   {validation_status} Plate detected: {plate_text} (conf: {conf_display})")
                        
                        # Log normalization details if available
                        if plate_result.get('normalization_details'):
                            norm = plate_result['normalization_details']
                            if norm.get('corrections_applied'):
                                logger.debug(f"   🔧 Corrections applied: {norm['corrections_applied']}")
                    else:
                        logger.info(f"   ❌ No plate detected")
"""

# 5. Patch for event logging (around line 700-720)
"""
Replace this section in the main processing loop:

OLD CODE:
                if action == 'enter':
                    plate_info = f" - {event.plate_text}" if event.plate_text != "Không nhận diện được biển" else ""
                    conf_info = f" (conf: {event.plate_confidence:.3f})" if event.plate_confidence > 0 else ""
                    logger.info(f"{sent_status} 🚗 {event.spot_name}: Vehicle entered{plate_info}{conf_info}")
                
                elif action == 'exit':
                    duration = result.get('duration_minutes', 0)
                    plate_info = f" - {event.plate_text}" if event.plate_text else ""
                    logger.info(f"{sent_status} 🚪 {event.spot_name}: Vehicle exited{plate_info} ({duration}m)")

NEW CODE:
                if action == 'enter':
                    log_plate_event('enter', event.spot_name, event.plate_text, event.plate_confidence, sent_status)
                elif action == 'exit':
                    duration = result.get('duration_minutes', 0)
                    plate_info = f" - {event.plate_text}" if event.plate_text else ""
                    duration_info = f" (duration: {duration}m)" if duration > 0 else ""
                    logger.info(f"{sent_status} 🚪 {event.spot_name}: Vehicle exited{plate_info}{duration_info}")
"""

# 6. Enhanced system status display
def format_system_status_display(system_status):
    """Format system status for better readability"""
    
    # Format occupancy info
    parking = system_status['parking_summary']
    occupancy_info = f"Occupied: {parking['occupied_spots']}/{parking['total_spots']} ({parking['occupancy_rate']}%)"
    
    # Format processing stats
    processing = system_status['processing_stats']
    detection_rate = processing.get('detection_rate', 0)
    validation_rate = processing.get('validation_rate', 0)
    
    processing_info = f"Events: {processing['events_sent']}/{processing['events_generated']}"
    plate_info = f"Plates: {processing['plates_detected']} detected ({detection_rate:.1f}%), {processing['plates_validated']} valid ({validation_rate:.1f}%)"
    
    # Format connection status
    conn = system_status['server_connection']
    conn_status = "Connected" if conn['is_connected'] else "Offline"
    queue_info = f", Queue: {conn['offline_queue_size']}" if conn.get('offline_queue_size', 0) > 0 else ""
    
    return {
        'occupancy': occupancy_info,
        'processing': processing_info,
        'plates': plate_info,
        'connection': f"Server: {conn_status}{queue_info}"
    }

# 7. Complete integration example
def apply_patches_to_main():
    """
    Instructions to apply these patches to main_parking_system.py:
    
    1. Add the helper functions at the top of the file (after imports)
    2. Replace the _process_vehicle_enter method sections as shown above
    3. Update the main processing loop event logging
    4. Optionally use enhanced system status formatting
    
    Key improvements:
    - Proper confidence percentage display (95.6% instead of 0.956310451%)
    - Better plate text selection (validated over raw)
    - Enhanced logging with validation status
    - Proper handling of double-line plates
    - Normalization correction logging
    """
    pass

# 8. Test the formatting functions
if __name__ == "__main__":
    print("=== Testing Plate Formatting Functions ===")
    
    # Test confidence formatting
    test_confidences = [0.9563104510307312, 0.85, 1.0, 0.0, 95.6]
    for conf in test_confidences:
        formatted = format_confidence_display(conf)
        print(f"Confidence {conf} → {formatted}")
    
    print("\n=== Testing Plate Text Selection ===")
    
    # Test plate text selection
    test_results = [
        {
            'text': '30F54504',
            'validated_text': '30F-545.04', 
            'confidence': 0.95,
            'is_valid': True
        },
        {
            'text': 'J0H79007',
            'validated_text': '30H-790.07',
            'confidence': 0.82,
            'is_valid': True
        },
        {
            'text': '30F\n54504',
            'validated_text': '30F\n54504',
            'confidence': 1.0,
            'is_valid': True
        },
        {
            'text': 'unclear_text',
            'validated_text': None,
            'confidence': 0.3,
            'is_valid': False
        }
    ]
    
    for i, result in enumerate(test_results):
        plate_text, confidence = get_best_plate_text(result)
        print(f"Test {i+1}: '{result['text']}' → '{plate_text}' (conf: {format_confidence_display(confidence)})")
    
    print("\n✅ Formatting tests completed!")