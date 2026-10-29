/*
 * FreshGuard ESP32 Multi-Sensor Transit & Spoilage Firmware
 * =========================================================
 * Hardware Configuration:
 *   - MQ-3 Gas Sensor (Analog ADC: GPIO 34 via voltage divider)
 *   - DHT22 Temp & Humidity (Digital: GPIO 4)
 *   - MPU-6050 6-Axis IMU (I2C: SDA=GPIO 21, SCL=GPIO 22)
 *   - Built-in / Warning LED (GPIO 2)
 *   - Optional Buzzer (GPIO 13)
 *
 * Capabilities:
 *   - 16-sample ADC oversampling for clean gas readings
 *   - High-speed 50Hz IMU sampling to capture transient drop/shock peaks
 *   - Pitch and roll inclination angles calculation
 *   - Automatic MPU-6050 fallback if disconnected
 *   - Bidirectional serial commands:
 *       "ALERT:ON"  -> Turns warning LED/buzzer on
 *       "ALERT:OFF" -> Turns warning LED/buzzer off
 *       "RESET_TRIP"-> Resets peak shock counter
 *
 * CSV Output Format (every 2 seconds):
 *   gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g,pitch_deg,roll_deg
 */

#include <Wire.h>
#include <DHT.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define MQ_PIN        34      // MQ-3 analog pin (GPIO34 is input-only ADC1)
#define DHT_PIN       4       // DHT22 data pin
#define DHT_TYPE      DHT22   // or DHT11
#define LED_PIN       2       // Onboard warning LED
#define BUZZER_PIN    13      // Actuator/Buzzer pin

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_MPU6050 mpu;
bool mpu_available = false;

// Sampling & Telemetry intervals
unsigned long last_transmit_ms = 0;
const unsigned long TRANSMIT_INTERVAL_MS = 2000; // Emit telemetry every 2 seconds

// IMU Shock and Orientation metrics
float peak_shock_g = 1.0;
float live_ax = 0.0, live_ay = 0.0, live_az = 1.0;
float live_pitch = 0.0, live_roll = 0.0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12); // ESP32 12-bit ADC (0 - 4095)

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  dht.begin();
  Wire.begin(21, 22); // I2C pins: SDA=21, SCL=22
  Wire.setClock(400000); // 400kHz Fast Mode I2C

  // Initialize MPU-6050
  if (mpu.begin()) {
    mpu_available = true;
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  } else {
    Serial.println("# WARNING: MPU-6050 not detected. Continuing in gas-only mode.");
  }

  delay(1200);
  // CSV Header
  Serial.println("gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g,pitch_deg,roll_deg");
}

void process_serial_commands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "ALERT:ON") {
      digitalWrite(LED_PIN, HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
    } else if (cmd == "ALERT:OFF") {
      digitalWrite(LED_PIN, LOW);
      digitalWrite(BUZZER_PIN, LOW);
    } else if (cmd == "RESET_TRIP") {
      peak_shock_g = 1.0;
    }
  }
}

void loop() {
  process_serial_commands();

  // High-frequency IMU loop (~50Hz) to capture instantaneous impacts & drops
  if (mpu_available) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Convert m/s^2 to standard Earth gravity (g)
    live_ax = a.acceleration.x / 9.80665;
    live_ay = a.acceleration.y / 9.80665;
    live_az = a.acceleration.z / 9.80665;

    // Instantaneous total acceleration vector magnitude
    float current_shock = sqrt(live_ax * live_ax + live_ay * live_ay + live_az * live_az);
    if (current_shock > peak_shock_g) {
      peak_shock_g = current_shock;
    }

    // Pitch and Roll calculation (in degrees)
    live_pitch = atan2(-live_ax, sqrt(live_ay * live_ay + live_az * live_az)) * 180.0 / PI;
    live_roll  = atan2(live_ay, live_az) * 180.0 / PI;
  }

  // Periodic Telemetry Transmission
  unsigned long now = millis();
  if (now - last_transmit_ms >= TRANSMIT_INTERVAL_MS) {
    last_transmit_ms = now;

    // 16-sample oversampling for MQ-3 analog noise reduction
    long gas_sum = 0;
    for (int i = 0; i < 16; i++) {
      gas_sum += analogRead(MQ_PIN);
      delay(2);
    }
    int gas_raw = gas_sum / 16;

    // Read DHT22
    float t = dht.readTemperature();
    float h = dht.readHumidity();

    if (!isnan(t) && !isnan(h)) {
      // Format: gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g,pitch_deg,roll_deg
      Serial.print(gas_raw);
      Serial.print(",");
      Serial.print(t, 1);
      Serial.print(",");
      Serial.print(h, 1);
      Serial.print(",");
      Serial.print(live_ax, 2);
      Serial.print(",");
      Serial.print(live_ay, 2);
      Serial.print(",");
      Serial.print(live_az, 2);
      Serial.print(",");
      Serial.print(peak_shock_g, 2);
      Serial.print(",");
      Serial.print(live_pitch, 1);
      Serial.print(",");
      Serial.println(live_roll, 1);
    }

    // Decay peak shock towards 1.0g for next interval
    peak_shock_g = 1.0;
  }

  delay(20); // 50Hz IMU polling cycle
}
