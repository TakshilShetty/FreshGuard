# FreshGuard ESP32 Sensor Firmware

Arduino C++ firmware for reading volatile gas concentration (MQ-3) and environmental metrics (DHT22) and streaming records over USB Serial.

## Hardware Specifications

- **Controller**: ESP32 Development Board (NodeMCU / ESP-WROOM-32)
- **Gas Sensor**: MQ-3 Alcohol / VOC Sensor (Analog Output AO)
- **Temperature & Humidity**: DHT22 (AM2302) or DHT11

## Circuit Diagram & Pinout

```text
ESP32 Dev Module
  │
  ├── 5V (VIN) ──────> MQ-3 VCC
  ├── 3.3V ──────────> DHT22 VCC
  ├── GND ───────────> Common GND (MQ-3 GND, DHT22 GND)
  │
  ├── GPIO34 (ADC) <── MQ-3 AO (via Voltage Divider: 10kΩ / 20kΩ to scale 5V -> 3.3V)
  └── GPIO4 (IO)   <── DHT22 DATA (with 4.7kΩ - 10kΩ pull-up to 3.3V)
```

> ⚠️ **Important Voltage Warning**: The ESP32 ADC pins operate at a maximum of 3.3V. Never connect the 5V MQ-3 analog output directly to GPIO34 without a voltage divider.

## Required Arduino Libraries

1. **DHT sensor library** by Adafruit (Version 1.4.0+)
2. **Adafruit Unified Sensor** by Adafruit

## Serial Protocol

The firmware continuously samples the sensors and outputs a CSV line every 2 seconds:
```text
gas_raw,temperature_c,humidity_percent
```

- **Baud Rate**: `115200`
- **gas_raw**: 12-bit ADC count (`0 - 4095`), averaged over 16 consecutive samples to attenuate electronic noise.
- **temperature_c**: Temperature in degrees Celsius (1 decimal precision).
- **humidity_percent**: Relative humidity % (1 decimal precision).
