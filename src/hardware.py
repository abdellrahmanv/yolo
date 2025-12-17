"""
Hardware Control Module - LCD and Buzzer for Raspberry Pi
LCD 16x2 I2C on GPIO 2/3, Buzzer on GPIO 18
"""

import time
import threading

# Try to import RPi.GPIO, mock if not on Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available (not on Pi?)")

# Try to import I2C LCD library
try:
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False
    print("Warning: RPLCD not available. Install: pip install RPLCD")


class BuzzerController:
    """
    Buzzer controller - beep frequency based on detection size
    Bigger box = faster beeping (like a parking radar)
    """
    
    def __init__(self, pin=18):
        self.pin = pin
        self.is_initialized = False
        self.beep_thread = None
        self.running = False
        self.current_frequency = 0  # Beeps per second
        
        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(self.pin, GPIO.OUT)
                GPIO.output(self.pin, GPIO.LOW)
                self.is_initialized = True
                print(f"Buzzer initialized on GPIO {self.pin}")
            except Exception as e:
                print(f"Buzzer init failed: {e}")
    
    def start(self):
        """Start the buzzer control thread"""
        if not self.is_initialized:
            return
        self.running = True
        self.beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
        self.beep_thread.start()
    
    def stop(self):
        """Stop buzzer"""
        self.running = False
        if self.is_initialized:
            GPIO.output(self.pin, GPIO.LOW)
    
    def set_frequency(self, beeps_per_second):
        """Set beep frequency (0 = no beeping)"""
        self.current_frequency = max(0, min(beeps_per_second, 20))  # Cap at 20 Hz
    
    def set_from_box_size(self, box_area, frame_area=320*320):
        """
        Set beep frequency based on bounding box size
        Bigger box = closer object = faster beeping
        
        Args:
            box_area: Area of bounding box in pixels
            frame_area: Total frame area (default 320x320)
        """
        if box_area <= 0:
            self.current_frequency = 0
            return
        
        # Calculate box ratio (0 to 1)
        ratio = box_area / frame_area
        
        # Map ratio to frequency:
        # - Very small (< 5%): 1 beep/sec
        # - Small (5-15%): 2-4 beeps/sec
        # - Medium (15-30%): 5-8 beeps/sec
        # - Large (30-50%): 10-15 beeps/sec
        # - Very large (> 50%): 20 beeps/sec (continuous)
        
        if ratio < 0.02:
            freq = 0  # Too small, probably noise
        elif ratio < 0.05:
            freq = 1
        elif ratio < 0.10:
            freq = 2
        elif ratio < 0.15:
            freq = 4
        elif ratio < 0.25:
            freq = 6
        elif ratio < 0.35:
            freq = 10
        elif ratio < 0.50:
            freq = 15
        else:
            freq = 20  # Very close
        
        self.current_frequency = freq
    
    def _beep_loop(self):
        """Background thread for beeping"""
        while self.running:
            try:
                if self.current_frequency > 0 and self.is_initialized:
                    # Beep on
                    GPIO.output(self.pin, GPIO.HIGH)
                    time.sleep(0.05)  # 50ms beep duration
                    GPIO.output(self.pin, GPIO.LOW)
                    
                    # Wait for next beep (safe division)
                    if self.current_frequency > 0:
                        interval = 1.0 / self.current_frequency - 0.05
                        if interval > 0:
                            time.sleep(interval)
                else:
                    time.sleep(0.1)  # Check again in 100ms
            except Exception:
                time.sleep(0.1)
    
    def cleanup(self):
        """Cleanup GPIO"""
        self.stop()
        if self.is_initialized:
            GPIO.output(self.pin, GPIO.LOW)


class LCDController:
    """
    LCD 16x2 I2C controller
    Shows glasses count and detection status
    """
    
    def __init__(self, i2c_address=0x27):
        self.lcd = None
        self.is_initialized = False
        self.last_message = ""
        
        if LCD_AVAILABLE:
            try:
                # Try common I2C addresses
                for addr in [0x27, 0x3F]:
                    try:
                        self.lcd = CharLCD(
                            i2c_expander='PCF8574',
                            address=addr,
                            port=1,
                            cols=16,
                            rows=2,
                            dotsize=8
                        )
                        self.is_initialized = True
                        print(f"LCD initialized at address 0x{addr:02X}")
                        self._show_startup()
                        break
                    except:
                        continue
                
                if not self.is_initialized:
                    print("LCD not found at 0x27 or 0x3F")
                    
            except Exception as e:
                print(f"LCD init failed: {e}")
    
    def _show_startup(self):
        """Show startup message"""
        if self.is_initialized:
            self.lcd.clear()
            self.lcd.write_string("Glasses Detector")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string("Starting...")
    
    def update(self, glasses_count, fps=0, status=""):
        """
        Update LCD display
        
        Line 1: Glasses: X    FPS
        Line 2: Status message
        """
        if not self.is_initialized:
            return
        
        try:
            # Line 1: Count and FPS
            line1 = f"Glasses: {glasses_count}"
            if fps > 0:
                line1 = f"{line1:11s}{fps:4.1f}"
            line1 = line1[:16].ljust(16)
            
            # Line 2: Status
            if glasses_count == 0:
                line2 = "No detection"
            elif glasses_count == 1:
                line2 = "1 pair detected"
            else:
                line2 = f"{glasses_count} pairs found"
            
            if status:
                line2 = status[:16]
            line2 = line2[:16].ljust(16)
            
            # Only update if changed
            message = line1 + line2
            if message != self.last_message:
                self.lcd.cursor_pos = (0, 0)
                self.lcd.write_string(line1)
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2)
                self.last_message = message
                
        except Exception as e:
            print(f"LCD update error: {e}")
    
    def show_message(self, line1, line2=""):
        """Show custom message"""
        if not self.is_initialized:
            return
        try:
            self.lcd.clear()
            self.lcd.write_string(line1[:16])
            if line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:16])
        except:
            pass
    
    def cleanup(self):
        """Cleanup LCD"""
        if self.is_initialized:
            try:
                self.lcd.clear()
                self.lcd.write_string("Goodbye!")
                time.sleep(0.5)
                self.lcd.clear()
                self.lcd.backlight_enabled = False
            except:
                pass


class HardwareManager:
    """
    Combined hardware manager for LCD and Buzzer
    """
    
    def __init__(self, buzzer_pin=18, lcd_address=0x27):
        print("Initializing hardware...")
        self.buzzer = BuzzerController(pin=buzzer_pin)
        self.lcd = LCDController(i2c_address=lcd_address)
        
    def start(self):
        """Start hardware controllers"""
        self.buzzer.start()
        
    def update_detection(self, detections, fps=0):
        """
        Update hardware based on detections
        
        Args:
            detections: List of detection dicts with 'bbox' key
            fps: Current FPS for display
        """
        count = len(detections)
        
        # Update LCD
        self.lcd.update(count, fps)
        
        # Update buzzer based on largest detection
        if count > 0:
            # Find largest bounding box
            max_area = 0
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                area = (x2 - x1) * (y2 - y1)
                max_area = max(max_area, area)
            
            self.buzzer.set_from_box_size(max_area)
        else:
            self.buzzer.set_frequency(0)
    
    def cleanup(self):
        """Cleanup all hardware"""
        print("Cleaning up hardware...")
        self.buzzer.cleanup()
        self.lcd.cleanup()
        if GPIO_AVAILABLE:
            GPIO.cleanup()


if __name__ == "__main__":
    print("Testing hardware...")
    
    hw = HardwareManager()
    hw.start()
    
    # Test LCD
    hw.lcd.show_message("Test Mode", "Hardware OK")
    time.sleep(2)
    
    # Test buzzer with increasing frequency
    print("Testing buzzer...")
    for freq in [1, 2, 4, 8, 10]:
        print(f"  Frequency: {freq} Hz")
        hw.buzzer.set_frequency(freq)
        time.sleep(1)
    
    hw.buzzer.set_frequency(0)
    hw.lcd.show_message("Test Complete!")
    time.sleep(1)
    
    hw.cleanup()
    print("Done!")
